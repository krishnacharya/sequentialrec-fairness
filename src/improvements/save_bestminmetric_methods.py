import os
import json
import sys
from pathlib import Path
from subprocess import check_call
from multiprocessing import Process
import multiprocessing
from tbparse import SummaryReader
from argparse import ArgumentParser
from src.utils.project_dirs import hparamdf_root
import pandas as pd

def filter_df(df, colname:str, field):
    return df[df[f'{colname}'] == field].reset_index(drop=True)

def lt_filter(df, lt_name:str):
    return df[df['loss_type'] == lt_name].reset_index(drop=True)

def pick_cp(df, cp_type):
    return df[df['valcp_type'] == cp_type].reset_index(drop=True)

def get_best_minnmetric(df, metric):
    '''
        metric e.g. R@20
        we want to sort by the corresponding val/bestmin_(metric) e.g. val/bestmin_R@20
    '''
    cp_type = f'bestavg_{metric}' # picking the avg metric used for checkpointing e.g.bestavg_R@20
    sort_key = f'val/bestmin_{metric}'
    s_df = pick_cp(df, cp_type).sort_values(by=sort_key, ascending=False, ignore_index=True) # first pick by the one used to checkpoint
    return s_df

def get_best_avgmetric(df, metric):
    '''
        metric e.g. R@20
        we want to sort by the val/bestavg_(metric) e.g. val/bestavg_R@20
    '''
    cp_type = f'bestavg_{metric}'
    sort_key = f'val/bestavg_{metric}'
    s_df = pick_cp(df, cp_type).sort_values(by=sort_key, ascending=False, ignore_index=True)
    return s_df

def combined_df_bestmethods(dfdict):
    '''
        the dataframes are sorted by best val min metric
    '''
    res = []
    for k,df in dfdict.items():
        res.append(df.loc[[0]]) # picking the first element, since sorted by val min metric
    return pd.concat(res, axis=0)

def percentimprovement_over_erm(df, metric_name):
    val = df[df['loss_type'] == 'erm'][metric_name]
    df[f'ermdelta_{metric_name}'] = (df[metric_name] - val) # delta can be negative
    df[f'ermreldelta_{metric_name}'] = (df[f'ermdelta_{metric_name}'] / val) # relative delta
    # df[f'ermimp_{metric_name}'] = (df[metric_name] - val) # improvement over erm
    # df[f'ermimp_percent_{metric_name}'] = 100 * (df[f'ermimp_{metric_name}'] / val)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--group", type=str, required=True)
    args = parser.parse_args()
    dirpath = hparamdf_root() / args.dataset/ args.group
    filepath = dirpath / "tbhpdash_all.pkl"   # FIXME hardcoded, assuming all res are saved with this pickle name
    if not filepath.exists():
        print("First generate the pkl file using src/utils/save_hparams_dash.py, with the appropriate dataset and group")
        return
    
    tbdashhp = pd.read_pickle(str(filepath))
    tbdashhp = tbdashhp[tbdashhp['valcp_type'].str.contains('avg', case=False, na=False)] # collecting only average val checkpointer
    lt = ['erm', 'cb','cb_log','ipw','ipw_log','group_dro', 's_dro','joint_dro']
    tb_lt = {k:lt_filter(tbdashhp, k) for k in lt}
    tb_bestmin_R20 = {k:get_best_minnmetric(tb_lt[k], 'R@20') for k in lt}
    tb_bestmin_nDCG20 = {k:get_best_minnmetric(tb_lt[k], 'nDCG@20') for k in lt}
    tb_bestavg_R20 = {k:get_best_avgmetric(tb_lt[k], 'R@20') for k in lt} # doesnt use group information in validation for collecting best method on groups etc
    tb_bestavg_nDCG20 = {k:get_best_avgmetric(tb_lt[k], 'nDCG@20') for k in lt}

    df_bestmin_R20 = combined_df_bestmethods(tb_bestmin_R20)
    df_bestmin_nDCG20 = combined_df_bestmethods(tb_bestmin_nDCG20)
    df_bestavg_R20 = combined_df_bestmethods(tb_bestavg_R20)
    df_bestavg_nDCG20 = combined_df_bestmethods(tb_bestavg_nDCG20)

    percentimprovement_over_erm(df_bestmin_R20, 'test/min_R@20')
    percentimprovement_over_erm(df_bestmin_nDCG20,'test/min_nDCG@20')
    percentimprovement_over_erm(df_bestavg_R20, 'test/min_R@20')
    percentimprovement_over_erm(df_bestavg_nDCG20,'test/min_nDCG@20')

    df_bestmin_R20.to_pickle(str(dirpath / "df_minR20_minval.pkl"))  # best minval R20 across training methods
    df_bestmin_nDCG20.to_pickle(str(dirpath / "df_minNDCG20_minval.pkl")) # best minval NDCG20 across training methods
    df_bestavg_R20.to_pickle(str(dirpath / "df_minR20_avgval.pkl"))
    df_bestavg_nDCG20.to_pickle(str(dirpath / "df_minNDCG20_avgval.pkl"))



if __name__ == "__main__":
    main()

