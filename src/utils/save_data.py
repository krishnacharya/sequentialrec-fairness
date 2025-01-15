import pandas as pd
import numpy as np

def save_mmap(actions: pd.DataFrame, save_dir, partition):
    '''
        Saved a pandas dataframe as mmap
        patition is either train, val, test
    '''
    data_np = actions.to_numpy(dtype="int32")
    data_len = len(actions)
    mmap = np.memmap(save_dir/f"{partition}.{data_len}.mmap", dtype="int32", shape = data_np.shape, mode = "w+")
    mmap[:] = data_np[:]
    mmap.flush()

def save_df(actions: pd.DataFrame, filepath):
    '''
        Saves a pandas dataframe as pickle
        parition is either train, val, test
    '''
    data_len = len(actions)
    actions.to_pickle(str(filepath))