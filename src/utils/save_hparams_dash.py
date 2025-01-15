import os
import json
import sys
from pathlib import Path
from subprocess import check_call
from multiprocessing import Process
import multiprocessing
from tbparse import SummaryReader
from argparse import ArgumentParser
from src.utils.project_dirs import dataset_group_hparamdf_dir

TB_S3_JOB_PATH = os.environ["TB_S3_JOB_PATH"]

def sync(jobs, logdir):
    for job in jobs:
        sync_cmd = ["aws", "s3", "sync", TB_S3_JOB_PATH.format(job), str(logdir/ (job + "/"))]
        print(" ".join(sync_cmd))
        check_call(sync_cmd)

def save_tb_hpdash_results(logdir, savepath, suffix = '_all'):
    reader = SummaryReader(str(logdir), pivot=True, extra_columns={'dir_name'})
    hparams_df = reader.hparams
    print('hparams_df shape, number of jobs', hparams_df.shape, hparams_df.shape[0]/8)

    scalars_df = reader.scalars
    test_cols = [c for c in scalars_df.columns if 'test' in c]
    test_res = scalars_df[scalars_df['step']==0] # only epoch zero in tensorboardlogs the test metrics
    final_test_res = test_res[['dir_name'] + test_cols].reset_index(drop=True)
    print('test metrics shape', final_test_res.shape, final_test_res.shape[0]/8)
  
    tbdash = hparams_df.merge(final_test_res, on = 'dir_name')
    print(tbdash.head())
    tbdash.to_csv(str(savepath / f"tbhpdash{suffix}.csv"), index=False)
    tbdash.to_pickle(str(savepath / f"tbhpdash{suffix}.pkl"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_filename", type=str, required=True) # finished job names
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    args = parser.parse_args()

    logdir = Path(__file__).parent / "logdir"
    check_call(["rm", "-rf", str(logdir)])
    logdir.mkdir()

    config_filename = args.config_filename
    if config_filename.endswith(".txt"):
        jobs = []
        for job in open(config_filename):
            print(job)
            jobs.append(job.strip())
    ctx = multiprocessing.get_context("spawn")
    sync_process = ctx.Process(target=sync, args=[jobs, logdir])
    sync_process.start()
    sync_process.join()

    save_tb_hpdash_results(logdir, dataset_group_hparamdf_dir(args.dataset, args.group)) # both of these are Pathlib variables