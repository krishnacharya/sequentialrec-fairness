import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import joint_dro
from src.recommenders.utils import joint_dro

import numpy as np
import pandas as pd
import itertools

class GroupInfo: # just used for LossComputerClass #FIXME refactor?
    def __init__(self, df, df_user_id_groups, lc_config):
        '''
            df could be train_actions, val_actions; schema is <user_id, item_id...>
            df_userid_groups a pandas dataframe each row has user_id to binary group mapping <0,1,0,..1.>
        '''
        num_sgroups = len(df_user_id_groups.attrs['sglist'])
        self.df_ugmap = None
        self.group_cols = None
        self.group_counts = None

        if lc_config.loss_type in ["joint_dro", "erm", "cb", "cb_log"]: # these dont use group info at all
            self.make_default_groupmap(df_user_id_groups, df) # doesnt matter, metric computer also measures only higher level group metrics

        elif lc_config.loss_type in ["s_dro", "group_dro", "ipw", "ipw_log"]:
            if num_sgroups == 1:
                self.make_default_groupmap(df_user_id_groups, df)
            elif num_sgroups == 2:
                if lc_config.subgroup_for_loss == 'atomic':
                    self.build_atomic_oh(df_user_id_groups, df)
                elif lc_config.subgroup_for_loss == '0':
                    gcolpick = df_user_id_groups.attrs['sglist'][0]
                    print(f'Using only {gcolpick}')
                    self.make_default_groupmap(df_user_id_groups, df, gcolpicklist=gcolpick)
                elif lc_config.subgroup_for_loss == '1':
                    gcolpick = df_user_id_groups.attrs['sglist'][1]
                    print(f'Using only {gcolpick}')
                    self.make_default_groupmap(df_user_id_groups, df, gcolpicklist=gcolpick)
                else:
                    print("subgroup_for_loss should only be 0, 1 or atomic")
                    raise AttributeError
            else:
                print("Currently expects upto 2 subgroups, can be extended to more")
                raise AttributeError
        else:
            raise AttributeError
    
    def make_default_groupmap(self, df_user_id_groups, df, gcolpicklist = None):
        if gcolpicklist is None: # FIXME fragile
            self.df_ugmap = df_user_id_groups
            self.group_cols = [col for col in df_user_id_groups.columns if '_g-' in col]
            self.n_groups = len(self.group_cols)
            self.group_counts = torch.LongTensor(df.merge(df_user_id_groups)[self.group_cols].sum(axis=0).values)
            print("training df group counts, rounding possible", df.shape, self.group_counts.sum(), self.group_counts)
        else:
            assert gcolpicklist is not None
            self.df_ugmap = df_user_id_groups[['user_id'] + gcolpicklist]
            self.group_cols = gcolpicklist
            self.n_groups = len(self.group_cols)
            self.group_counts = torch.LongTensor(df.merge(df_user_id_groups)[self.group_cols].sum(axis=0).values)
            print("training df group counts, rounding possible", df.shape, self.group_counts.sum(), self.group_counts)
    
    def build_atomic_oh(self, df_uid_groups, df):
        '''
            df_uid_groups here is the userid_gmap original <0,1,0 | 1,0,0 | 1,0>, shape # number of user_id x num groups
            converts intersecting subgroups <0,1,0 | 1,0,0 | 1,0> type to
            one hot of length = len of subgroup1 x len subgroup 2 x len subgroup3
        '''
        def df_to_oh_atomic(df) -> tuple[np.array, np.array]:
            '''
                df binary elements has n x total original subgroups(intersecting)
                for e.g. for popcold: n x 6 original [[niche,div,pop], [small,med,long]]
            '''
            subgroup_arrays = [df[group_cols].values for group_cols in subgroups]
            active_indices = np.array([np.argmax(arr, axis=1) for arr in subgroup_arrays]).T
            atomic_indices = np.dot(active_indices, multipliers)
            one_hot_encoded = np.zeros((len(df), total_groups), dtype=np.float32)
            one_hot_encoded[np.arange(len(df)), atomic_indices] = 1
            return one_hot_encoded, atomic_indices
        
        def get_vector_ai_gname() -> pd.DataFrame:
            '''
                for each combos of <1,0,0 | 1,0,0 | 1,0> the corresponding atomic oh index and atomic group name
            '''
            def get_all_vectors(): # all combos of  <1,0,0 | 1,0,0 | 1,0> with permutaions within |
                slist = []
                for group_cols in subgroups:
                    slist.append(tuple([1] + [0]*(len(group_cols)-1)))
                permutations_list = [set(itertools.permutations(s)) for s in slist]
                cartesian_product = itertools.product(*permutations_list)
                all_combos = np.array([tuple(itertools.chain.from_iterable(tup)) for tup in cartesian_product]) 
                assert len(all_combos) == len(np.unique(all_combos, axis=0)) == total_groups
                return all_combos
            vall = get_all_vectors()
            gcols_np = np.array(all_gcols) # just converting to numpy for easy indexing in the v in vall loop below
            vaig_df = pd.DataFrame(vall, columns = all_gcols)
            _, atomic_indices = df_to_oh_atomic(vaig_df)
            atomic_names = []
            for v in vall:
                atomic_names.append("atom_g-" + "-".join([ele.split('_g-')[1] for ele in gcols_np[v.astype(bool)]])) #FIXME fragile string processing to get intersecting group name #FIXME fragile
            vaig_df['atomic_gnames'] = atomic_names
            vaig_df['atomic_indices'] = atomic_indices
            return vaig_df.sort_values(by = ['atomic_indices'])
        
        # all_gcols = [col for col in df_uid_groups.columns if '_g-' in col] #FIXME fragile 
        all_gcols = df_uid_groups.attrs['glist'] # original group columns
        subgroups = df_uid_groups.attrs['sglist']
        group_sizes = np.array([len(group) for group in subgroups], dtype=np.int32)
        total_groups = np.prod(group_sizes) #total number of atomic groups
        multipliers = np.cumprod(np.concatenate(([1], group_sizes[:-1])))

        vaig_df = get_vector_ai_gname()
        oh_atomic, _ = df_to_oh_atomic(df_uid_groups) # big dataframe

        df_ugmap_temp = pd.DataFrame(oh_atomic, columns = vaig_df.atomic_gnames.values)
        df_ugmap_temp['user_id'] = df_uid_groups['user_id']
        self.make_default_groupmap(df_ugmap_temp, df)

class LossComputerConfig(object):
    def __init__(self, loss_type = 'erm', adj=0, gdro_stepsize=0.01, stream_lr = 0.1,
                 streaming_gloss_epochreset = False,
                 normalize_loss=False, joint_dro_alpha=0.2, subgroup_for_loss='0'):
        self.loss_type = loss_type
        self.joint_dro_alpha = joint_dro_alpha
        #below are related to group_dro variants
        self.adj = adj
        self.gdro_stepsize = gdro_stepsize
        self.stream_lr = stream_lr
        self.streaming_gloss_epochreset = streaming_gloss_epochreset
        self.normalize_loss = normalize_loss
        self.subgroup_for_loss = subgroup_for_loss

class LossComputer:
    def __init__(self, loss_type='erm',adj:float=0.0,
        gdro_stepsize=0.01, normalize_loss=False, 
        stream_lr = 0.1, streaming_gloss_epochreset=False,
        joint_dro_alpha=None, group_names:list[str] = None,
        data_group_counts = None, subgroup_for_loss = '0'
    ):
        assert loss_type in ["s_dro", "group_dro", "erm", "joint_dro", "ipw", "ipw_log", "cb", "cb_log"]
        self.loss_type = loss_type
        self.gdro_stepsize = gdro_stepsize # eta_g
        self.stream_lr = stream_lr
        self.streaming_gloss_epochreset = streaming_gloss_epochreset
        self.normalize_loss = normalize_loss
        self.group_names = group_names
        self.n_groups = len(group_names)
        self.cbloss = False
        self.cbsmooth = False
        self.subgroup_for_loss = subgroup_for_loss

        if data_group_counts is not None:       # used only by ipw and gdro actual losses
            self.glob_group_counts = data_group_counts.cuda()
            self.group_frac = self.glob_group_counts / self.glob_group_counts.sum() #used in greedy robust loss FIXME  group_frac variable is overloaded! rename; self.group_grac is group_frac in the whole dataset; also clashes with update_stats but not a problem
            self.ipweights = self.glob_group_counts.sum() / self.glob_group_counts # inverse propensity weights
            self.log_ipweights = torch.log(self.ipweights)
        
        if self.loss_type == "joint_dro":
            # Joint DRO reg should be 0.
            assert joint_dro_alpha is not None
            self._joint_dro_loss_computer = joint_dro.RobustLoss(joint_dro_alpha, 0, "cvar") # size of uncertainty set, strength of regularizer, geometry cvar

        if loss_type == "group_dro":
            self.adj = torch.tensor(adj).float().cuda() # same adjustment C for all groups!
        
        if loss_type == "cb":
            self.cbloss = True
            self.cbsmooth = False
        
        if loss_type == "cb_log":
            self.cbloss = True
            self.cbsmooth = True

        # quantities below are maintained throughout training, not reset every epoch
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups # should never have require grad true
        self.streaming_gloss = torch.zeros(self.n_groups).cuda()
        # self.exp_avg_loss = torch.zeros(self.n_groups).cuda() # only used for btl method
        # self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()
        self.reset_stats()
    
    def loss(self, per_sample_loss, group_onehot=None):
        '''
            persample loss has shape (batchsize)
            group_onehot is a binary tensor of shape [Batchsize, #groups]
            
            returns  actual loss is the loss thats used for backpropagating
            
            update stats
            group loss is a tensor of shape # groups that the the loss on each group
            group count is just the number of examples belonging to each group in this batch of examples
            weights is only for gDRO and has the weights we maintain on each group

        '''
        # compute per-sample and per-group losses
        group_loss, group_count = self.compute_group_avg(per_sample_loss, group_onehot) #  avg loss for each group in this batch
        # compute overall loss
        if self.loss_type == "group_dro":
            actual_loss = self.compute_robust_loss(group_loss)
            # else: #deprecated btl method
            #     self.update_exp_avg_loss(group_loss.detach(), group_count.detach())
            #     actual_loss, weights = self.compute_robust_loss_btl(group_loss)
        elif self.loss_type == "s_dro":
            actual_loss = self.compute_streaming_robustloss(group_loss)
        elif self.loss_type == "joint_dro":
            actual_loss = self._joint_dro_loss_computer(per_sample_loss) # doesnt use group info
        elif self.loss_type in {"erm", "cb", "cb_log"}:
            actual_loss = per_sample_loss.mean()
        elif self.loss_type == 'ipw': #inverse weighting prop to global group size in training
            actual_loss = self.compute_ipw_loss(group_loss, group_count)
        elif self.loss_type == 'ipw_log': #inverse weighting prop to global group size in training
            actual_loss = self.compute_ipw_loss(group_loss, group_count, use_logweight=True)
        else:
            raise NotImplementedError
        
        # update stats detaching, memory leaks in og code
        self.update_stats(actual_loss.detach(), group_loss.detach(), group_count.detach(), per_sample_loss.detach().mean())
        return actual_loss # this is used backpropagated

    def compute_ipw_loss(self, group_loss, group_count, use_logweight=False):
        '''
            computes  1/Batchsize * sum_g (totaltrainsize/freq_g train) sum_i=1^B  indicator(example i is group g) * loss example i 
        '''
        if use_logweight:
            return (group_loss * group_count * self.log_ipweights).sum() / group_count.sum() # denominator is always batch size e.g 32, 64, 128
        else:
            return (group_loss * group_count * self.ipweights).sum() / group_count.sum() # denominator is always batch size e.g 32, 64, 128
        
    def compute_group_avg(self, losses, group_onehot):
        '''
            Compute the loss for each group, essentially just collect Average CE loss for each group
            returns group_loss which has shape (n_groups), group_count has number of disjoint groups also shape (n_groups) with sum = batch size
            group_loss will have requires gradient, group_count will not
        '''
        # compute observed counts and mean loss for each group
        group_map = group_onehot.T.cuda() # has shape (#groups, batchsize)
        group_count = group_map.sum(1) # shape #groups, number of datapoints belonging to each of the groups in this batch
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        return (group_map @ losses.view(-1)) / group_denom, group_count

    def compute_robust_loss(self, group_loss):
        '''
            Basically for each group find its loss in this batch of examples
            sum w_g group_loss_g, also updates w_g based on Sagawa DRO

            Returns 
                sum w_g group_loss_g, weights on each group
                robust loss and self.adv_probs

                self.adv_probs will not require gradients
        '''
        adjusted_loss = group_loss # shape is (#number of groups)
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.glob_group_counts) # TODO check generalization adjustment from original gDRO paper
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        # this is not using a streaming group loss, uses the loss current batch for a group
        self.adv_probs = self.adv_probs * torch.exp(self.gdro_stepsize * adjusted_loss.detach()) # see Alg 1 in gDRO Sagawa, exp(stepsize * loss for that group)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum()) # normalize and get weights on each group

        robust_loss = group_loss @ self.adv_probs
        # return robust_loss, self.adv_probs
        return robust_loss
    
    def compute_streaming_robustloss(self, group_loss): # s-dro
        Lg_tilde = (1 -self.stream_lr) * self.streaming_gloss + self.stream_lr * group_loss.detach() # line 6 in google dro recsys paper
        self.adv_probs = self.adv_probs * torch.exp(self.gdro_stepsize * Lg_tilde) # Lgtilde no gradients
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        srobust_loss = group_loss @ self.adv_probs #check scalar straming rate factor in front?
        return srobust_loss
    
    def reset_stats(self): #reset at the end of each epoch
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        # self.update_data_counts = torch.zeros(self.n_groups).cuda()
        # self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_ps_loss = 0.0
        self.avg_actual_loss = 0.0 # this is the loss used for backprop, e.g w_g loss on group g, CVAR loss etc...
        self.batch_count = 0.0
        if self.streaming_gloss_epochreset:
            self.streaming_gloss = torch.zeros(self.n_groups).cuda()

    def update_stats(self, actual_loss, group_loss, group_count, psl_batchmean):
        '''
            logging info, ensure all are detached tensors to avoid mem leaks
            updated after every batch
        '''
        # avg group loss just using the fact that elementwise newavg = old x (n/n+m) + current x (m/n+m)
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss
        self.streaming_gloss = (1-self.stream_lr) * self.streaming_gloss  + self.stream_lr * group_loss

        # batch-wise average actual loss
        denom = self.batch_count + 1 # HACK avoid using `denom` as tensor above and scalar now
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss + (1 / denom) * actual_loss # robust loss or just erm ...
        self.avg_ps_loss = (self.batch_count / denom) * self.avg_ps_loss + (1 / denom) * psl_batchmean
        
        # update counts
        self.processed_data_counts += group_count
        self.batch_count += 1

        # if self.loss_type == "group_dro":
        #     self.update_data_counts += group_count * ((weights > 0).float()) #TODO what use are these?
        #     self.update_batch_counts += ((group_count * weights) > 0).float()
        # else:
        #     self.update_data_counts += group_count
        #     self.update_batch_counts += (group_count > 0).float()
        
        # avg per-sample quantities, overall quantities
        # group_frac = self.processed_data_counts / (self.processed_data_counts.sum()) # still calculates correctly with intersecting group
        # self.avg_per_sample_loss = group_frac @ self.avg_group_loss #FIXME self.avg_per_sample_loss not correct when groups are intersecting
        # if not torch.allclose(self.avg_ps_loss, self.avg_per_sample_loss, atol = 0.01):
        #     print(self.avg_ps_loss, self.avg_per_sample_loss)


    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx, gname in enumerate(self.group_names):
            stats_dict[f"avg_loss_group:{gname}"] = self.avg_group_loss[idx].item()
            # stats_dict[f"exp_avg_loss_group:{idx}"] = self.exp_avg_loss[
            #     idx].item()
            # stats_dict[f"processed_data_count_group:{gname}"] = self.processed_data_counts[idx].item()
            # stats_dict[f"update_data_count_group:{gname}"] = self.update_data_counts[idx].item()
            # stats_dict[f"update_batch_count_group:{gname}"] = self.update_batch_counts[idx].item()
        stats_dict["avg_actual_loss"] = self.avg_actual_loss.item() # actual loss is used for backprop
        stats_dict["avg_per_sample_loss"] = self.avg_ps_loss.item()
        return stats_dict

    # TODO below isnt used, alternative gDRO methods
    # def update_exp_avg_loss(self, group_loss, group_count):
    #     prev_weights = (1 - self.gamma * (group_count > 0).float()) * (self.exp_avg_initialized > 0).float() 
    #     curr_weights = 1 - prev_weights
    #     self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
    #     self.exp_avg_initialized = (self.exp_avg_initialized >0) + (group_count > 0)

    # def compute_robust_loss_btl(self, group_loss): # be the leader? 
    #     adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(self.glob_group_counts)
    #     return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    # def compute_robust_loss_greedy(self, group_loss, ref_loss):
    #     sorted_idx = ref_loss.sort(descending=True)[1]
    #     sorted_loss = group_loss[sorted_idx]
    #     sorted_frac = self.group_frac[sorted_idx]

    #     mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
    #     weights = mask.float() * sorted_frac / self.alpha
    #     last_idx = mask.sum()
    #     weights[last_idx] = 1 - weights.sum()
    #     weights = sorted_frac * self.min_var_weight + weights * (1 - self.min_var_weight)

    #     robust_loss = sorted_loss @ weights

    #     # sort the weights back
    #     _, unsort_idx = sorted_idx.sort()
    #     unsorted_weights = weights[unsort_idx]
    #     return robust_loss, unsorted_weights

    def get_model_stats(self, model, args, stats_dict): # just some L2 loss related
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param)**2
        stats_dict["model_norm_sq"] = model_norm_sq.item()
        stats_dict["reg_loss"] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict