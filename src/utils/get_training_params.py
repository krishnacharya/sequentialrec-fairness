from src.sagemaker.sagemaker_utils import is_running_on_sagemaker
import json
import os 
from argparse import ArgumentParser
import torch
import numpy as np

def set_seed(seed):
    """Sets seed, do this before starting any training, dataload/splits"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(s):
     if s  in ["True", "true", "1", 1, "yes", "y"]:
        return True
     elif s in ["False", "false", "0", 0, "no", "n"]:
        return False
     else:
         raise AttributeError(f"can parse boolean variable from string '{s}'")
    
def get_training_parameters(params_descr):
    if is_running_on_sagemaker():
        training_env = json.loads(os.environ["SM_TRAINING_ENV"])
        parameters = training_env["hyperparameters"]
        job_name = training_env["job_name"] 
        for name, type, default in params_descr:
            new_name = name.replace("-", "_")
            if name in parameters:
                if type != bool:
                    parameters[new_name] = type(parameters[name])
                else:
                        parameters[new_name] = str2bool(parameters[name])
            else:
                parameters[new_name] = default
        parameters["job_name"] = job_name
    else:
        arg_parser = ArgumentParser() 
        for name, t, default in params_descr:
            if t != bool:
                arg_parser.add_argument("--" + name, type=t, default=default)
            else:
                arg_parser.add_argument("--" + name, type=str2bool, default=default)
        args = arg_parser.parse_args() 
        parameters = vars(args) 
        parameters["job_name"] = None 
    return parameters