from argparse import ArgumentParser
from pathlib import Path
import dill
import pandas as pd
import torch
import tqdm
from src.utils.load_data import load_data
from src.utils.project_dirs import *
from src.eval.generate_local_irm import gen_local_irm
from src.recommenders.recommender import Recommender
from ir_measures import nDCG, R
import tempfile
import os
from subprocess import check_call

TB_S3_JOB_PATH = os.environ["TB_S3_JOB_PATH"]

def save_irm(testdata:pd.DataFrame, dataset:str, group:str, metric:str, job_name, topk=20): # processes single job
    jobs_path = os.environ["SAGEMAKER_DEFAULT_BUCKET"] + "/" + job_name
    model_s3_path = jobs_path + "/output/model.tar.gz"
    print(f"Model s3 path: {model_s3_path}")

    workdir = tempfile.mkdtemp(prefix=job_name + "_")
    print("local work dir: ", workdir)
    model_output_path = Path(workdir) / "output"
    model_output_path.mkdir(parents=True)

    check_call(["aws", "s3", "cp", model_s3_path, str(model_output_path)])
    check_call(["tar", "xzvf", str(model_output_path/"model.tar.gz"), "-C", model_output_path])
    checkpoint = [f for f in model_output_path.iterdir() if f.suffix == ".dill"][0]
    recommender:Recommender = dill.load(open(checkpoint, "rb"))
    gen_local_irm(testdata, dataset, group, metric, recommender, job_name, topk=topk)
    check_call(["rm", "-rf", str(workdir)])

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True) #ml1m..
    parser.add_argument("--group", type=str, required=True) # pop33,pop2060,..,popseq
    parser.add_argument("--metric", type=str, required=True, help="one of NDCG20 or R20, used to select the dataframe in hparamdf, df_minNDCG20_avgval or df_minR20_avgval") # both use average checkpoint but sorted by avg NDCG@20 or avg R@20
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()
    if args.metric not in ['nDCG@20', 'R@20']:
        raise AttributeError
    namemap = {'nDCG@20': 'NDCG20', 'R@20':'R20'}
    testdata = load_data(args.dataset)['test']
    df_best = pd.read_pickle(dataset_group_hparamdf_dir(dataset=args.dataset, group=args.group) / f'df_min{namemap[args.metric]}_avgval.pkl')
    job_names = [v.split('/')[0] for v in df_best['dir_name'].values]
    for job_name in job_names:
        save_irm(testdata, args.dataset, args.group, args.metric, job_name, args.topk)

if __name__ == "__main__":
    main()