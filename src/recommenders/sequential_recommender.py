import io
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time
from src.recommenders.sequential_model import SequentialModel
from src.recommenders.recommender import Recommender
from src.recommenders.utils.sequencer  import Sequencer
from src.recommenders.utils.checkpointer import CheckpointMonitor
from ir_measures import nDCG, R
from torch.utils.data import DataLoader
from src.recommenders.utils.logger import TrainingLogger
from copy import deepcopy
import torch
import tqdm
from PIL import Image
import ir_measures
from collections import defaultdict, deque

from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run
from src.recommenders.utils.loss import LossComputer, LossComputerConfig, GroupInfo
from src.recommenders.utils.metric_computer import MetricComputer

class SequentialRecommenderConfig(object):
    def __init__(self, sequence_len=200, batch_size=512, device='cpu:0', 
                max_epoch=100000, batches_per_epoch=128, 
                val_checkpointing_metric=nDCG@10, 
                val_logging_metrics = [nDCG@10, R@10, nDCG@20, R@20],
                val_topk=20,
                early_stop_patience_epochs=200,
                repetitions_filter = False, 
                effective_batch_size = 512,
                use_weighted_loss = False,
                new_loss_weight = 0.9,
                randomfrac_train = False,
                early_stop_on = "overall",
                log_tradeoff_trajectory=False
                ):
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.device = device 
        self.max_epochs = max_epoch 
        self.batches_per_epoch = batches_per_epoch
        self.val_checkpointing_metric = val_checkpointing_metric
        self.val_logging_metrics = val_logging_metrics
        self.val_topk = val_topk
        self.early_stop_patience_epochs = early_stop_patience_epochs
        self.repetitions_filter = repetitions_filter
        self.effective_batch_size = effective_batch_size
        self.use_weighted_loss = use_weighted_loss
        self.new_loss_weight = new_loss_weight
        self.early_stop_on = early_stop_on
        self.log_tradeoff_trajectory = log_tradeoff_trajectory
        self.randomfrac_train = randomfrac_train
        
def plot_to_image(func):
  def wrapper(*args, **kwargs):
        figure = func(*args, **kwargs)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = Image.open(buf)
        image = np.array(image)
        image = image / 255.0  
        image = np.transpose(image, (2, 0, 1))
        return image
  return wrapper


@plot_to_image
def plot_tradeoff_trajectory(trajectory, tradeoff_name):
    n = len(trajectory)
    indices = list(range(n))  # Original indices
    
    # Ensure the trajectory has at most 1000 points while preserving the first and last point
    if n > 1000:
        step = n // 1000
        trajectory = [trajectory[i] for i in range(0, n, step)]
        indices = [indices[i] for i in range(0, n, step)]
        if trajectory[-1] != trajectory[n-1]:
            trajectory.append(trajectory[n-1])
            indices.append(n-1)
    
    metric1_values = [x[0] for x in trajectory]
    metric2_values = [x[1] for x in trajectory]
    
    # Create a color map from dark green to red
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a scatter plot instead of a line plot
    scatter = ax.scatter(metric1_values, metric2_values, c=indices, cmap=cmap, s=10)

    # Increase size of the last point and mark it in pure red
    ax.scatter(metric1_values[-1], metric2_values[-1], color='red', marker='o', s=100)

    metric1_name = tradeoff_name.split(':::')[0]
    metric2_name = tradeoff_name.split(':::')[1]
    ax.set_xlabel(metric1_name)
    ax.set_ylabel(metric2_name)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Step in trajectory', rotation=270, labelpad=15)
    
    ax.grid()
    return fig


def add_tb_log_eoe(tblogger:TrainingLogger, stats:dict, epoch:int, prefix:str): # HACK y rn 
    ''' 
        Adds to tensorboard at the end of epoch
        prefix : train, val
    '''
    for key, value in stats.items():
        if 'exp_' in key or '0_set' in key: # latter is for the boolean is set
            continue
        elif 'loss' in key:
            name = '_loss_'
        elif 'epoch' in key:
            name = 'epoch'
        elif 'patience' in key:
            name = 'patience'   
        elif 'nDCG' in key:
            name = '_nDCG_'
        elif 'R@' in key:
            name = '_recall_'
        else:
            continue
        if ('g-' in key):
            tblogger.add_scalar(name = prefix + name + 'group/' + key, value=value , step=epoch)
        elif 'best' in key:
            tblogger.add_scalar(name = prefix + name + 'best/' + key, value=value , step=epoch)
        else:
            tblogger.add_scalar(name = prefix + name + 'overall/' + key, value=value , step=epoch)


def last_measure_is_pareto_optimal(enu_metric_history, rnu_metric_history):
    last_enu, last_rnu = enu_metric_history[-1], rnu_metric_history[-1]
    optimal = True
    for enu, rnu in zip(enu_metric_history[:-1], rnu_metric_history[:-1]):
        if enu >= last_enu and rnu >= last_rnu:
            optimal = False
    return optimal

class SequentialRecommender(Recommender):
    def __init__(self, config: SequentialRecommenderConfig) -> None:
        self.sequencer:Sequencer = None # TODO val actions added at the end of full training! ok for from scratch, be careful when retraining
        self.model: SequentialModel = None #lazy init with init_model, TODO can add pretrained sequential model here! for retraining an ERM tuned model on validation 
        self.config: SequentialRecommenderConfig = config
        if config.effective_batch_size % config.batch_size != 0:
            raise AttributeError("Effective batch size should be a multiple of batch size")
        self.batches_per_update = self.config.effective_batch_size // self.config.batch_size
        assert self.batches_per_update == 1 #for now
        self.batches_per_epoch = self.config.batches_per_epoch * self.batches_per_update
        self.monitor = None  # initialize while running train_refac

    def init_model(self, sequencer: Sequencer) -> SequentialModel:
        raise NotImplementedError
    
    def run_eval(self, val_actions, loss_computer:LossComputer=None, metric_computer:MetricComputer=None):
        '''
            update loss computer with the validation loss and returns a list of recommendations
            topk for each user_id in the val_actions
        '''
        self.model.eval()
        val_userids = list(val_actions.user_id) # this is of size N_VAL user, also each element is unique, see the corresponding src/preprocess/ for this dataset
        val_itemids = list(val_actions.item_id) # this will also have size N_VAL users, but as some movies can obviously repeat
        start = 0
        recs = [] # topk recommendations for each user list of list of tuples, each tuple is <item id, score>
        with torch.no_grad():
            while start < len(val_itemids):
                batch_userids = val_userids[start:start+self.config.batch_size]
                batch_itemids = val_itemids[start:start+self.config.batch_size]
                batch = self.sequencer.get_sequences_batch(batch_userids, randomfrac=False)
                valt = [self.sequencer.item_mapping.get(item_id, self.sequencer.unknown_item_id) for item_id in batch_itemids] #  tokenized validation targets items for the users above, mapping using sequencer, with default value of unknown if never seen item in training
                gt_items = torch.tensor(valt, device=self.config.device, dtype=torch.int64).unsqueeze(-1) # shape (batchsize x 1), validation targets
                unknown_item_mask  = (gt_items != int(self.sequencer.unknown_item_id)).to(torch.float32).squeeze(-1) # shape batchsize, 1 if known item
                batch_on_device = {k:v.to(self.config.device) for (k,v) in batch.items()}
                logits = self.model.predict(batch_on_device)
                logprobs = logits.log_softmax(-1).gather(1, gt_items).squeeze(-1) # get logsoftmax for val item, shape is (batchsize)
                psl = -logprobs * unknown_item_mask  # basically we dont count loss for never seen items; shape (batchsize)
                loss_computer.loss(per_sample_loss = psl, group_onehot = batch['groups']) # FIXME not correct psl class balanced loss!

                # setting special tokens to -infty to not appear in topk
                logits[:, :self.sequencer.num_special_items] = -float("inf") # -infty for special tokens, padding unknown etc
                if self.config.repetitions_filter: # if true means there are no repetitions in user watch data, so -infty to those items
                    logits = self.sequencer.repetitions_filter(batch_userids, logits)
                top_k_recs = torch.topk(logits, self.config.val_topk) # tuple with max scalars and item ids

                # gt_logprobs.append(logprobs.squeeze(-1)) 
                # gt_is_predictable.append(unknown_item_mask.detach().squeeze(-1)) # basically we dont count loss for never seen items, 1 if seen item
                decoded_recs = self.sequencer.decode_topk(top_k_recs)
                start += self.config.batch_size
                recs += decoded_recs
            # END full pass thru validation data
            metric_computer.update_metrics(recs) # val metric stats updated

    def run_train_epoch(self, optimizer, loader, loss_computer = None):
        self.model.train()
        pbar = tqdm.tqdm(total=self.batches_per_epoch, ncols=70, ascii=True)
        batches_processed = 0
        while batches_processed < self.batches_per_epoch: # TODO wonder if lr decay etc is impacted nyway step is done after each batch?
            for batch in loader:
                if batches_processed >= self.batches_per_epoch:
                    break
                batch_on_device = {k:v.to(self.config.device) for (k, v) in batch.items()}
                result = self.model.forward(batch_on_device, cbloss = loss_computer.cbloss, cbsmooth=loss_computer.cbsmooth) # FIXME fragile
                loss = loss_computer.loss(per_sample_loss = result['loss_per_sample'], group_onehot = batch['groups'])  # update_stats() within loss_computer will find average group loss etc
                loss.backward()
                batches_processed += 1
                if batches_processed % self.batches_per_update == 0: # usually self.batches_per_update is 1 so this is standard
                    optimizer.step()
                    optimizer.zero_grad()
                pbar.update(1)
        pbar.close()
    
    def train_refac(self, train_actions, val_actions, df_userid_groups, tensorboard_dir, lc_config:LossComputerConfig) -> dict:
        '''
            df_userid_groups a pandas dataframe each row has user_id to binary group indicators <0,1,0,0,0,0>, 
            if more than one group is active gdro loss method contructs groups for every atomic 
            <0,1,0, 1,0,0>; subgroups: popviewer, seqlen -> 9 hot for training loss
            with no duplicate rows, unique user_id

            be careful with df_userid_groups, if it has more than one group active then we run into trouble when 
            TODO check calculation of avg_per_sample_loss in update stats of LossComputer,  average metrics(NDCG, Recall) 
            will be still be correct because its handled by irmeasures
        '''
        torch.autograd.set_detect_anomaly(True)
        group_ob = GroupInfo(train_actions, df_userid_groups, lc_config) # can detect if its atomic groups or not, if not it provides a lambda to map sequencers groups to atomic
        self.sequencer = Sequencer(train_actions, group_ob, max_length=self.config.sequence_len, randomfrac_default=self.config.randomfrac_train)
        self.model = self.init_model(self.sequencer)  # TODO change here if using pretrained ERM model
        logger = TrainingLogger(tensorboard_dir)
        all_user_ids = train_actions.user_id.unique()
        train_loader = DataLoader(all_user_ids, collate_fn=self.sequencer.get_sequences_batch, batch_size=self.config.batch_size, shuffle=True) # different batches can have diff sequence lengths, collator ensures that within a batch the sequences are same size
        self.monitor = CheckpointMonitor(self.config)
        print(f"Gradient accumulation for {self.batches_per_update} batches")
        
        optimizer = torch.optim.Adam(self.model.parameters())
        train_loss_computer = LossComputer(loss_type=lc_config.loss_type, joint_dro_alpha=lc_config.joint_dro_alpha,
                                           adj=lc_config.adj, gdro_stepsize=lc_config.gdro_stepsize, 
                                          streaming_gloss_epochreset = lc_config.streaming_gloss_epochreset, stream_lr  = lc_config.stream_lr,
                                          group_names=group_ob.group_cols, data_group_counts=group_ob.group_counts,
                                          subgroup_for_loss=lc_config.subgroup_for_loss)
        
        val_loss_computer = LossComputer(loss_type=lc_config.loss_type, joint_dro_alpha=lc_config.joint_dro_alpha,
                                         adj=lc_config.adj, gdro_stepsize=lc_config.gdro_stepsize,
                                        streaming_gloss_epochreset = lc_config.streaming_gloss_epochreset, stream_lr  = lc_config.stream_lr, 
                                        group_names=group_ob.group_cols, data_group_counts=group_ob.group_counts,
                                        subgroup_for_loss=lc_config.subgroup_for_loss)
        
        #still using original groups dataframe here for calculation metrics, we dont want to measure metrics on atomic
        val_metric_computer = MetricComputer(val_actions, df_userid_groups, metrics = self.config.val_logging_metrics) #checkpointing and saving based on this metric
        
        for epoch in range(1, self.config.max_epochs+1):
            print(f"==== epoch: {epoch} ====")
            # training epoch
            self.run_train_epoch(optimizer = optimizer, loader=train_loader, loss_computer = train_loss_computer)
            train_loss_stats = train_loss_computer.get_stats() # average group loss, actual loss
            add_tb_log_eoe(tblogger=logger, stats = train_loss_stats, epoch = epoch, prefix="train")

            # validation
            self.run_eval(val_actions = val_actions, loss_computer = val_loss_computer, metric_computer = val_metric_computer)
            val_loss_stats = val_loss_computer.get_stats()
            self.monitor.update(val_metric_computer.results, epoch, self.model) #handles checkpointing boilerplate
            add_tb_log_eoe(tblogger=logger, stats = val_loss_stats, epoch = epoch, prefix = 'val')
            add_tb_log_eoe(tblogger=logger, stats = val_metric_computer.results, epoch = epoch, prefix = 'val') # logs epoch to epoch group metrics
            add_tb_log_eoe(tblogger=logger, stats = self.monitor.get_bestepochs_patience(), epoch = epoch, prefix = 'val') # logs best avgval_metric, best minval_metric
            logger.tb_writer.flush()
            if self.monitor.patience <= 0:
                print("No patience left. Early stopping after epoch {epoch}")
                break
            val_loss_computer.reset_stats()
            train_loss_computer.reset_stats()

        self.sequencer.add_val_actions(val_actions)
        del(train_loader)
        return val_metric_computer.get_metrics(split_name='val') # picks up only best avg and best min

    def recommend(self, user_ids, top_k, model_weight, batch_size=None): # called only at the end for test recommendations
        if batch_size is None:
            batch_size = self.config.batch_size
        self.model.load_state_dict(model_weight)
        self.model.eval()
        result = []
        with torch.no_grad():
            start = 0
            while start < len(user_ids):
                batch_userids = user_ids[start:start+batch_size]
                batch = self.sequencer.get_sequences_batch(batch_userids, randomfrac=False, shortening=1)
                batch_on_device = {k:v.to(self.config.device) for (k,v) in batch.items()}
                logits = self.model.predict(batch_on_device)
                logits[:, :self.sequencer.num_special_items] = -float("inf")
                if self.config.repetitions_filter:
                    logits = self.sequencer.repetitions_filter(batch_userids, logits)
                top_k_recs = torch.topk(logits, top_k)
                decoded_recs = self.sequencer.decode_topk(top_k_recs)
                result += decoded_recs
                start += batch_size
        return result