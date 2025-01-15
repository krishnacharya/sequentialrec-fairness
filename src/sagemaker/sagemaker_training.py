import copy
import datetime
import json
import uuid

import boto3
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch
from pathlib import Path
from src.utils.project_dirs import get_src_dir, dataset_splits_dir, project_root, uid_to_group_dir
import tempfile
import os
from subprocess import check_call
import sagemaker
import sys

def copy_tree(src: Path, dst:Path):
    dst.mkdir(exist_ok=True, parents=True)
    print(f"copying {src} to {dst}")
    check_call(["cp", "-r", str(src), str(dst)])

def run_sagemaker_training(entry_point, dataset, parameters, config_name, num_days=14):
    '''
        References for Sagemaker PyTorch estimator and it's parent
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
        https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework
    '''
    sagemaker_iam_role = os.environ["SAGEMAKER_IAM_ROLE"]
    sagemaker_s3_bucket = os.environ["SAGEMAKER_OUTPUT_BUCKET"]
    sagemaker_region = os.environ.get("SAGEMAKER_REGION", "us-west-2")
    instance_type = "ml.p3.2xlarge"

    job_id = datetime.datetime.now().strftime('%y%j%H%M%S')
    config_name_fixed = config_name.replace("_", "-").replace(".", "-").replace(",", "-")
    job_name = f"{config_name_fixed}-{job_id}"

    sagemaker_dir = Path(tempfile.mkdtemp(prefix=f"repeatbump_sagemaker_{dataset}_{Path(entry_point).stem}_"))

    data_dir = dataset_splits_dir(dataset)
    rel_data_path = data_dir.relative_to(project_root())
    dst_data_dir = sagemaker_dir / (rel_data_path.parent)
    
    data_gmap_dir = uid_to_group_dir(dataset) # user to gmap datafram directory
    real_gmap_path = data_gmap_dir.relative_to(project_root())
    dst_gmap_dir = sagemaker_dir / (real_gmap_path.parent)

    print("sagemaker dir:", sagemaker_dir)
    copy_tree(get_src_dir(), sagemaker_dir) 
    copy_tree(data_dir, dst_data_dir)
    copy_tree(data_gmap_dir, dst_gmap_dir) # copying the uid_to_group directory
    copy_tree(project_root()/"requirements.txt", sagemaker_dir)
    print("done copying")
    tb_s3_output = sagemaker_s3_bucket + "/tensorboard"
    parameters["dataset"] = dataset
    print("creating the estimator...")
    estimator = PyTorch(
        source_dir=str(sagemaker_dir),
        entry_point=str(entry_point),
        role=sagemaker_iam_role,
        instance_type=instance_type,
        instance_count=1,
        volume_size=200,
        base_job_name="test",
        framework_version="2.2.0",
        keep_alive_period_in_seconds = 3600,
        hyperparameters=parameters,
        environment={"TQDM_DISABLE": "1"},
        tensorboard_output_config=TensorBoardOutputConfig(s3_output_path=tb_s3_output),
        py_version="py310",
        max_run=num_days*24*3600,
        sagemaker_session=sagemaker.session.Session(boto3.Session(region_name=sagemaker_region)))
    print("running the job...")
    estimator.fit(job_name=job_name, wait=False)
    print("done")
    return job_name

def combinations(hyperparameters):
    if len(hyperparameters) == 0:
        yield hyperparameters
    else:
        first_key = list(hyperparameters.keys())[0]
        first_key_values = hyperparameters[first_key]
        hp_copy =  copy.deepcopy(hyperparameters)
        del(hp_copy[first_key])
        for first_key_val in first_key_values:
            for other_param_vals in combinations(hp_copy):
                res = copy.deepcopy(other_param_vals)
                res[first_key] = first_key_val
                yield res


def is_possible(parameters, impossible_combinations):
    for combination in impossible_combinations:
        is_possible = False
        for k, v in combination.items():
            if parameters[k] != v:
                is_possible = True
        if not is_possible:
            return False
    return True
    

def process_config(config_name, training_config):
    entry_point = training_config["entry_point"] 
    dataset = training_config["dataset"]
    num_days = training_config.get("num_days", 14) #default 2 weeks runtime for a process, use account with this quota
    parameters = training_config["parameters"]
    impossible_combinations = training_config.get("impossible_parameter_combinations", [])

    hyperparameters = training_config.get("hyperparameters", {})
    jobname_rename = training_config.get("jobname_rename", {}) 
    job_names = []
    for hp in combinations(hyperparameters):
        all_params = parameters | hp 
        if is_possible(all_params, impossible_combinations):
            job_suffix = "-".join([f"{jobname_rename.get(k, k)}-{v}" for (k, v) in hp.items()])
            full_config_name = config_name + "-" + job_suffix
            print(full_config_name, all_params)
            job_name = run_sagemaker_training(entry_point, dataset, all_params, full_config_name, num_days=num_days)
            job_names.append(job_name)
        else:
            print("ignoring impossible configuration", all_params)
    print("Following jobs were launched")
    print("\n".join(job_names))

if __name__ == "__main__":
    train_configs_dir = Path(__file__).parent/"training_configs"
    train_configs_dir.mkdir(exist_ok=True)
    config_name = Path(sys.argv[1]).stem
    config_path = Path(sys.argv[1])
    with open(config_path, "r") as config_file:
        training_config = json.load(config_file)

    if (type(training_config) == dict):
        process_config(config_name, training_config)

    elif (type(training_config) == list):
        for sub_config in training_config:
            process_config(config_name, training_config)
