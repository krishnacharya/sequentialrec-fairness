from argparse import ArgumentParser
import json
from pathlib import Path
import dill
import ir_measures
import pandas as pd
from src.utils.load_data import load_data, load_ugmap_df
from src.utils.project_dirs import get_dataset_checkpoints_dir, get_recs_dir
from src.recommenders.recommender import Recommender
from ir_measures import nDCG, R
from src.recommenders.utils.metric_computer import MetricComputer
from copy import deepcopy

def test_recommender(checkpoint_path:str, dataset:str,
                         run_output_path:str, data, df_userid_groups, recommender, 
                         metrics = [nDCG@10, R@10, nDCG@20, R@20], top_k=20):
    '''
        data: test data <userid, itemid>
        df_userid_groups: user_id to gmap which may be intersecting
    '''
    def get_result_savejson(checkpoint_name:str) -> dict:
        def result_to_json(mw_dict):
            for cp_type, mw in mw_dict.items():
                recs = recommender.recommend(user_ids, top_k, mw)
                test_metric_computer.update_metrics(recs)
                result = test_metric_computer.get_onerun_testmetrics()
                name = checkpoint_name + '_' + cp_type
                result["Model"] = name
                metrics_file_path = recs_folder / (f"{name}.json")
                jsonable_result = {str(k): v for (k, v) in result.items()}
                with open(metrics_file_path, "w") as out:
                    json.dump(jsonable_result, out, indent=4)
                print(f"{cp_type} mw processed")
        user_ids = list(data.user_id)
        result_to_json(recommender.monitor.best_minmetric_modelweight)
        result_to_json(recommender.monitor.best_avgmetric_modelweight)
        # mw_keys = set(recommender.monitor.best_avgmetric_modelweight.keys() + recommender.monitor.best_minmetric_modelweight.keys())
        # for k in mw_keys if

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise AttributeError(f"recommender not saved {checkpoint_path}")
    if run_output_path == None:
        recs_folder = get_recs_dir(dataset)
    else:
        recs_folder = Path(run_output_path)
        recs_folder.mkdir(exist_ok=True, parents=True)
    test_metric_computer = MetricComputer(data = data, df_userid_groups=df_userid_groups, metrics = metrics)
    get_result_savejson(checkpoint_path.stem)