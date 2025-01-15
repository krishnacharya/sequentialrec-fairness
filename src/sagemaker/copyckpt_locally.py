from argparse import ArgumentParser
from pathlib import Path
from subprocess import check_call
from src.utils.project_dirs import get_recs_dir
from subprocess import Popen
import tempfile

import os

parser = ArgumentParser()

parser.add_argument("--job-name", required=True)
args = parser.parse_args()

workdir = tempfile.mkdtemp(prefix=args.job_name + "_")
print("local work dir: ", workdir)
jobs_path = os.environ["SAGEMAKER_DEFAULT_BUCKET"] + "/" + args.job_name

sources_path = jobs_path + "/source/sourcedir.tar.gz"
check_call(["aws", "s3", "cp", sources_path, workdir]) #just copying the sourcedir tar from the bucket and extracting in temp directory, no trained model here
check_call(["tar", "xzvf", workdir + "/sourcedir.tar.gz", "-C", workdir])
data_path = Path(workdir) / "data/processed"
dataset  = [f for f in data_path.iterdir()][0].stem # will be string ml1m, rr etc
model_output_path = Path(workdir) / "output"/ "checkpoints" / dataset
model_output_path.mkdir(parents=True)
model_s3_path = jobs_path + "/output/model.tar.gz"
print(f"Model s3 path: {model_s3_path}") # has the saved recommender, which has sequential model, monitors member variables...

check_call(["aws", "s3", "cp", model_s3_path, str(model_output_path)])
check_call(["tar", "xzvf", str(model_output_path/"model.tar.gz"), "-C", model_output_path]) # uncompress model tar.gz
checkpoint = [f for f in model_output_path.iterdir() if f.suffix == ".dill"][0]