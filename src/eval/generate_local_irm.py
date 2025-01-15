from argparse import ArgumentParser
import json
from pathlib import Path
import dill
import ir_measures
import pandas as pd
import torch
import tqdm
from src.utils.load_data import load_data
from src.utils.project_dirs import get_dataset_group_metric_irmdir
from src.utils.ir_measures_converters import get_irmeasures_qrels, get_irmeasures_run
from src.recommenders.recommender import Recommender
from ir_measures import nDCG, R

def gen_local_irm(test_data:pd.DataFrame, dataset:str, group, metric,
                   recommender, jobname, topk=20):
    '''
        metric: nDCG@20 or R@20
    '''
    user_ids = list(test_data.user_id)    
    irm_dir = get_dataset_group_metric_irmdir(dataset, group, metric)
    print(jobname)
    irmeasures_run_file = irm_dir / (f"{jobname}_top_{topk}_irm_run.csv")
    if not irmeasures_run_file.exists():
        model_weight = recommender.monitor.best_avgmetric_modelweight[f'bestavg_{metric}']
        recs = recommender.recommend(user_ids, topk, model_weight, batch_size=1024)
        run: pd.DataFrame = get_irmeasures_run(recs, test_data)
        run.to_csv(irmeasures_run_file, index=False)
    else:
        run = pd.read_csv(irmeasures_run_file)
        run["query_id"] = run.query_id.astype('str')
        run["doc_id"] = run['doc_id'].astype('str')