import pandas as pd
import numpy as np
from src.utils.project_dirs import raw_data_root, processed_data_root
from collections import defaultdict
from train_test_split import save_mmap
import requests
from numpy.random import default_rng

DATA_FILE = raw_data_root()/ "ml1m/ml-1m.txt" 
GROUP_DIR = processed_data_root() / "ml1m"/ "groups"
SPLIT_DIR = processed_data_root() / "ml1m"/"split"

N_VAL_USERS = 6040

def main():
    '''
        LOO dataset split, train, val and test
    '''
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(exist_ok=True, parents=True)
        response = requests.get("https://github.com/FeiSun/BERT4Rec/blob/master/data/ml-1m.txt")
        data = response.text
        DATA_FILE.write_text(data)

    user_items = defaultdict(list) 
    for line in open(DATA_FILE):
        user, item = line.strip().split()
        user = int(user)
        item = int(item)
        user_items[user].append(item)

    train = []
    val = []
    test = []
    all_users = sorted(list(user_items.keys()))
    rng = default_rng(42) # TODO config file for randomness for validation
    val_users = rng.choice(all_users, N_VAL_USERS, replace=False)

    for user  in user_items:
        is_val_user = user in val_users 
        sequence = user_items[user]
        timestamps = range(1, len(sequence) + 1)
        for i in range(len(sequence)):
            item = sequence[i]
            timestamp = timestamps[i]
            triplet = (user, item, timestamp)
            if (i < len(sequence) - 2) or ((not is_val_user) and i == len(sequence) -2):
                train.append(triplet)
            elif ((is_val_user) and i == len(sequence) -2):
                val.append(triplet)
            else:
                assert(i == len(sequence) -1)
                test.append(triplet)
    train = pd.DataFrame(train, columns=['user_id', 'item_id', 'timestamp']) 
    test = pd.DataFrame(test, columns=['user_id', 'item_id', 'timestamp'])
    val = pd.DataFrame(val, columns=['user_id', 'item_id', 'timestamp'])
    SPLIT_DIR.mkdir(exist_ok=True, parents=True)
    save_mmap(train, SPLIT_DIR, "train")
    save_mmap(val, SPLIT_DIR, "val")
    save_mmap(test, SPLIT_DIR, "test")

if __name__ == "__main__":
    main()
