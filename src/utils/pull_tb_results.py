import json
import sys
from tensorboard import program
from pathlib import Path
from subprocess import check_call
from multiprocessing import Process
import multiprocessing
import tensorboard
import os
import time

TB_S3_JOB_PATH = os.environ["TB_S3_JOB_PATH"]

def sync(jobs, logdir):
    while True:
        for job in jobs:
            sync_cmd = ["aws", "s3", "sync", TB_S3_JOB_PATH.format(job), str(logdir/ (job + "/"))]
            print(" ".join(sync_cmd))
            check_call(sync_cmd)
        time.sleep(10)
        
if __name__ == "__main__":
    logdir = Path(__file__).parent / "logdir" 
    check_call(["rm", "-rf", str(logdir)])
    logdir.mkdir() # makes logdir in utils
    config_filename = sys.argv[1]
    if config_filename.endswith(".json"):
        jobs = json.load(open(sys.argv[1], "r"))
    elif config_filename.endswith(".txt"):
        jobs = []
        for job in open(config_filename):
            print(job)
            jobs.append(job.strip())
    print("monitring jobs: \n\t", "\n\t".join(jobs))
    tb = tensorboard.program.TensorBoard()
    tb.configure(argv=[None, '--logdir', str(logdir), '--host', '127.0.0.1'])
    url = tb.launch()
    print(f"tensorboard is listening on {url}")

    ctx = multiprocessing.get_context("spawn")
    sync_process = ctx.Process(target=sync, args=[jobs, logdir])
    sync_process.start()
    sync_process.join()