from datetime import datetime
import json
from src.recommenders.SASRec import SASRecRecommender, SASRecConfig
import dill
import gc
import torch

from src.utils.load_data import load_data, load_ugmap_df
from src.sagemaker.sagemaker_utils import  sagemaker_tb_dir, is_running_on_sagemaker, sagemaker_model_output_dir, sagemaker_data_output_dir
from src.utils.project_dirs import get_dataset_checkpoints_dir, get_model_dataset_tensorboard_dir, get_recs_dir, get_src_dir
from src.utils.get_training_params import get_training_parameters, set_seed
from src.recommenders.utils.logger import TrainingLogger
from subprocess import check_call
from src.recommenders.utils.loss import LossComputerConfig

from src.utils.validate_config import get_config_dict
from src.eval.eval_groups_bestmodel import test_recommender

RAND_SEED = 42

parameters = get_training_parameters((("dataset", str, "retailrocket_views"), 
                                      ("device", str, "cuda:0"),
                                      ("max-batches-per-epoch", int, 128),
                                      ("repetitions-filter", bool, False),
                                      ("max-epochs", int, 2),
                                      ("batch-size", int, 256),
                                      ("effective-batch-size", int, 256),
                                      ("early-stop-on", str, "none"),
                                      ("val-checkpointing-metric", str, "nDCG@20"),
                                      ("early-stop-patience-epochs", int, 200),
                                      ("sequence-len", int, 200),
                                      ("dropout", float, 0.5),
                                      ("embedding-size", int, 256),
                                      ("num-layers", int, 3),
                                      ("dim-feedforward", int, 256),
                                      ("nhead", int, 1),
                                      ("loss-type", str, "joint_dro"),
                                      ("joint-dro-alpha", float, 0.33), #cvar alpha
                                      ("gdro-stepsize", float, 0.01), # gdro step size
                                      ("adj", float, 0),
                                      ("stream-lr", float, 0.1),
                                      ("streaming-gloss-epochreset", bool, False),
                                      ("groups", str, "popseq_bal"),
                                      ("subgroup-for-loss", str, "0") # by default just use subgroup 0, other options ["0", "1", "atomic"]
                                      ))

if parameters["job_name"] is not None:
    job_id = parameters["job_name"]
else:
    job_id = datetime.now().strftime('%Y%m%d%H%M%S')

set_seed(RAND_SEED)
data_df = load_data(dataset = parameters['dataset'])
df_userid_groups = load_ugmap_df(dataset = parameters['dataset'], groups = parameters["groups"])
model_name = f"SASRec_{parameters['dataset']}_{job_id}"

if is_running_on_sagemaker():
    tensorboard_dir = sagemaker_tb_dir()
    output_dir = sagemaker_model_output_dir() 
    data_output_dir = sagemaker_data_output_dir()

else:
    tensorboard_dir = get_model_dataset_tensorboard_dir(parameters['dataset'], model_name)
    output_dir = get_dataset_checkpoints_dir(parameters['dataset'])
    data_output_dir = get_recs_dir(parameters["dataset"])

print("Model Training parameters:", parameters)
print("Tensorboard dir:", tensorboard_dir)
print("Output dir:", output_dir)

out_file =  output_dir/ f"{model_name}.dill"

config = SASRecConfig(device=parameters['device'], 
                      batches_per_epoch=parameters['max_batches_per_epoch'],
                      repetitions_filter=parameters['repetitions_filter'], 
                      max_epoch=parameters["max_epochs"], 
                      batch_size =parameters["batch_size"],
                      effective_batch_size =parameters["effective_batch_size"],
                      dropout=parameters["dropout"],
                      embedding_size=parameters["embedding_size"],
                      num_layers=parameters["num_layers"],
                      dim_feedforward=parameters["dim_feedforward"],
                      nhead = parameters["nhead"],
                      sequence_len = parameters["sequence_len"],
                      early_stop_on = parameters["early_stop_on"],
                      early_stop_patience_epochs = parameters["early_stop_patience_epochs"]
                    )

lc_config = LossComputerConfig(loss_type = parameters['loss_type'],
                               joint_dro_alpha = parameters['joint_dro_alpha'],
                               gdro_stepsize = parameters['gdro_stepsize'], # step size for probability for each group
                               adj = parameters['adj'],
                               stream_lr = parameters['stream_lr'],
                               streaming_gloss_epochreset = parameters['streaming_gloss_epochreset'],
                               subgroup_for_loss = parameters['subgroup_for_loss'] # used for getting the groups used by IPW, gDRO and sDRO in the sequencer, see GroupInfo for how it maps to group one hot
                               )

hp_dict = get_config_dict(config)
hp_dict.update(get_config_dict(lc_config))

def train():
    recommender = SASRecRecommender(config)
    valmetrics = recommender.train_refac(data_df['train'], data_df["val"], df_userid_groups, tensorboard_dir, lc_config)
    hp_dict.update(valmetrics)
    with open(out_file, "wb") as out:
        print(f"saving recommender obj to {out_file}")
        dill.dump(recommender, out)
    # del(recommender)
    gc.collect()
    torch.cuda.empty_cache()
    return recommender

logger = TrainingLogger(tensorboard_dir)
recommender = train()

#Test metrics, # TODO change here, df_userid_groups just during test evaluation
test_recommender(checkpoint_path = str(out_file.absolute()), dataset = parameters["dataset"],
                    run_output_path = str(data_output_dir), data = data_df['test'], 
                    df_userid_groups = df_userid_groups, recommender=recommender, top_k=20,
                    metrics = config.val_logging_metrics)

def add_tboard_hparams(mw_suffixes:list[str]):
    hp_dict.update({'valcp_type' : 'none'}) #HACK y
    for suffix in mw_suffixes:
        with open(data_output_dir / (out_file.stem + f'_{suffix}.json')) as metrics_file: # used to be out_file.stem
            metrics_dict = json.load(metrics_file)
        metrics = {}
        for metric_name, metric_val in metrics_dict.items():
            if type(metric_val) in [float, int]:
                metrics[metric_name] = metric_val
        hp_dict['valcp_type'] = suffix
        logger.add_hparams(hp_dict, metrics) # can only log float and int in values

# adding to tensorboard hparam
add_tboard_hparams(recommender.monitor.best_avgmetric_modelweight.keys())
add_tboard_hparams(recommender.monitor.best_minmetric_modelweight.keys())