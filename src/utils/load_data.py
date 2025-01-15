from pathlib import Path
import pandas as pd
import numpy as np
import json

def set_str_dtype(df):
    df.user_id = df.user_id.astype("str")
    df.item_id = df.item_id.astype("str")
    return df


def load_data(dataset):
    dataset_dir = Path(__file__).absolute().parent.parent.parent / "data/processed" / dataset / "split"
    result = {}
    for file in dataset_dir.iterdir():
        partition, size = file.stem.split(".")
        size = int(size)
        if  partition in ["train"]:
            data = np.memmap(file, dtype="int32", mode="r", shape=(size, 3))
            df = pd.DataFrame(data, columns=["user_id", "item_id", "timestamp"])
            result[partition] = set_str_dtype(df)
        elif partition in ["val", "test"]:
            data = np.memmap(file, dtype="int32", mode="r", shape=(size, 3))
            df = pd.DataFrame(data, columns=["user_id", "item_id", "timestamp"])
            result[partition] = set_str_dtype(df) 
    return result

def load_ugmap_df(dataset:str, groups:str=''):
    '''
        get the user id to group mapping, group can be cold start based, pop viewer based or both combined
    '''
    dir = Path(__file__).absolute().parent.parent.parent / "data/processed" / dataset / "uid_to_group"
    return pd.read_pickle(str(dir) + f'/{groups}.pkl')

def load_saved_json(json_path:str) -> dict:
    '''
        json_path Absolute path to json file
    '''
    with open(json_path) as json_data:
        res = json.load(json_data)
    return res