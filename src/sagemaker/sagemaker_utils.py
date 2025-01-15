import json
import os
from pathlib import Path


def sagemaker_ml_dir():
    return Path("/opt/ml/")

def sagemaker_tb_dir():
    tb_dir = sagemaker_ml_dir() / "output/tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tb_dir

def sagemaker_checkpoints_dir():
    checkpoints_dir = sagemaker_ml_dir() / "output/checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir

def sagemaker_model_output_dir():
    dir_name = sagemaker_ml_dir() / "model"
    dir_name.mkdir(parents=True, exist_ok=True)
    return dir_name

def sagemaker_data_output_dir():
    dir_name = sagemaker_ml_dir() / "data"
    dir_name.mkdir(parents=True, exist_ok=True)
    return dir_name

def get_experiment_name():
        training_env = json.loads(os.environ["SM_TRAINING_ENV"])
        parameters = training_env["hyperparameters"]
        return "RepeatBump-" + parameters["dataset"]

def get_run_name():
        training_env = json.loads(os.environ["SM_TRAINING_ENV"])
        return training_env["job_name"]




def is_running_on_sagemaker():
    return sagemaker_ml_dir().exists()