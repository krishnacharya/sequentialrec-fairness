#Time-based split for sequential recommendation
import pandas as pd
from pathlib import Path
from numpy.random import default_rng
from collections import Counter, defaultdict
import numpy as np

from src.utils.project_dirs import raw_data_root, processed_data_root
from src.utils.save_data import save_mmap

def ensure_sorted(actions):
    timestamps = list(actions.timestamp)
    for i  in range(len(timestamps) - 1):
        if timestamps[i+1] < timestamps[i]:
            return False
    return True

DATA_DIR = Path("__file__").absolute().parent / "data" / "processed"

def core5_sort_LOOsplitsave(df, dataset_name:str, rand_seed=42, n_val_users=10000):
    '''
        df must have user_id, timestamp and item_id columns
        first core 5 for users and items
        followed by sorting by <userid, timestamp, int> break ties in timestamp, int
        LOO split of this dataframe and save into splits directory as train, val, test mmap
    '''
    output_rawdir = raw_data_root() / dataset_name # this is where to store the txt file with <user id , item id> interactions
    output_rawdir.mkdir(parents=True, exist_ok=True)
    
    df = core_k(df, 5)
    print("Shape after core 5", df.shape)
    df = df.sort_values(by = ['user_id', 'timestamp', 'item_id'])
    with open(output_rawdir / "all_core5_sorted.txt", 'w') as f:
        df[["user_id", "item_id"]].to_string(f, index=False, header=False, justify = 'left')

    txt_filepath = output_rawdir / "all_core5_sorted.txt"
    LOO_split(txt_filepath, dataset_name, rand_seed=rand_seed, n_val_users=n_val_users)

# def sort_core5_save(df:pd.DataFrame, dataset_name:str, \
#                     rand_seed = 42, n_val_users = 10000):
#     '''
#         sort by (user_id, timestamp int), then core 5 for users and item and save
#         df must have user_id, timestamp and item_id columns
#     '''
#     output_rawdir = raw_data_root() / dataset_name
#     output_rawdir.mkdir(parents=True, exist_ok=True)
    
#     df = df.sort_values(by = ['user_id', 'timestamp', 'item_id'])
#     data_core5 = core_k(df, 5)
#     print("Shape after core 5", data_core5.shape)

#     data_core5.to_csv(output_rawdir / "all_core5.csv", index=False)
#     with open(output_rawdir / "all_core5.txt", 'w') as f:
#         data_core5[["user_id", "item_id"]].to_string(f, index=False, header=False, justify = 'left')
    
#     txt_filepath = output_rawdir / "all_core5.txt"
#     LOO_split(txt_filepath, dataset_name, rand_seed = rand_seed, n_val_users = n_val_users)


def get_first_action(actions) -> pd.DataFrame:
    first_action_idx = actions.groupby('user_id').timestamp.idxmin()
    first_actions = actions.loc[first_action_idx].reset_index(drop=True)
    return first_actions

def cnt_filter(actions, k, field):
    field_counts = Counter(actions[field]).most_common()
    valid_vals = {x[0] for x in list(filter(lambda x: x[1] >=k, field_counts))}
    actions_filtered = actions[actions[field].isin(valid_vals)]
    return actions_filtered

 # leave only users who interacted with k _different_ items and items that have been interacted by 5 different users
def core_k(actions, k):
    actions_deduplicated = actions.drop_duplicates(subset = ["user_id", "item_id"])
    original_size = len(actions_deduplicated)
    print(f"core-k filtering. original size:{original_size}")
    i = 1 
    while True:
        start_len = len(actions_deduplicated)
        print(f"iteration {i}.")
        actions_deduplicated = cnt_filter(actions_deduplicated,k, "user_id")
        print(f"size after users_filter: {len(actions_deduplicated)}")

        actions_deduplicated = cnt_filter(actions_deduplicated, k, "item_id")
        print(f"size after items_filter: {len(actions_deduplicated)}")
        i += 1 
        end_len = len(actions_deduplicated)
        if (end_len == start_len):
            break
    selected_users = set(actions_deduplicated['user_id'])
    selected_items = set(actions_deduplicated['item_id']) 
    print(f"numer of users after filtering: {len(selected_users)}")
    print(f"numer of items after filtering: {len(selected_items)}")
    filtered_by_user = actions[actions.user_id.isin(selected_users)]
    filtered_by_item = filtered_by_user[filtered_by_user.item_id.isin(selected_items)]
    print(f"dataset size after core-{k} filter: {len(filtered_by_item)}")
    return filtered_by_item

#We might transfer files over network, so make sure that the files will be stored in the most efficient way
# def save_mmap(actions: pd.DataFrame, save_dir, partition):
#     data_np = actions.to_numpy(dtype="int32")
#     data_len = len(actions)
#     mmap = np.memmap(save_dir/f"{partition}.{data_len}.mmap", dtype="int32", shape = data_np.shape, mode = "w+")
#     mmap[:] = data_np[:]
#     mmap.flush()

# def save_dfgroups(actions: pd.DataFrame, save_dir, partition:str, ngroups:int):
#     data_len = len(actions)
#     actions.to_pickle(save_dir/f"{partition}.{data_len}.{ngroups}.pkl")

def split_data(dataset, test_fraction=0.1, val_fraction=0.01, n_val_users=10000, k_core_filter=5, rand_seed=42):
    filename = DATA_DIR /  dataset / "all.csv"
    all_actions = pd.read_csv(filename)
    if not ensure_sorted(all_actions):
        raise BrokenPipeError(f"{filename} contains non-sorted actions")
    all_actions = core_k(all_actions, k_core_filter)

    test_borderline_idx = int(len(all_actions) * (1 - test_fraction))
    test_borderline = all_actions.iloc[test_borderline_idx].timestamp

    val_borderline_idx  = int(len(all_actions) * (1 - (test_fraction + val_fraction)))
    val_borderline =  all_actions.iloc[val_borderline_idx].timestamp

    train_actions = all_actions[all_actions.timestamp < val_borderline]
    all_val_actions = all_actions[all_actions.timestamp.between(val_borderline, test_borderline, inclusive='left')] 

    
    test_actions = all_actions[all_actions.timestamp > test_borderline] 


    all_train_users_counts = Counter(train_actions.user_id).most_common()
    all_train_users = set()
    for user, cnt in all_train_users_counts:
        if cnt < k_core_filter:
            break
        all_train_users.add(user)

    all_val_actions = all_val_actions[all_val_actions.user_id.isin(all_train_users)]
    test_actions =  test_actions[test_actions.user_id.isin(all_train_users)] 

    all_val_user_ids = all_val_actions.user_id.unique() 
    if len(all_val_user_ids) > n_val_users: 
        rng = default_rng(rand_seed)
        val_users = rng.choice(sorted(list(all_val_user_ids)), n_val_users, replace=False) 
    else:
        val_users = all_val_user_ids

    val_actions = all_val_actions[all_val_actions.user_id.isin(set(val_users))] 
    val_actions = get_first_action(val_actions)
    test_actions = get_first_action(test_actions)

    train_val_actions = pd.concat([train_actions, all_val_actions])
    train_itemsets = train_actions.groupby("user_id")["item_id"].agg(set=lambda x: set(x))['set'].to_dict()
    train_val_itemsets = train_val_actions.groupby("user_id")["item_id"].agg(set=lambda x: set(x))['set'].to_dict()
    val_is_rep = []
    for user_id, item_id  in val_actions[["user_id", "item_id"]].itertuples(index=False):
        val_is_rep.append(int(item_id in train_itemsets[user_id]))
    val_actions["is_repetition"] = val_is_rep

    test_is_rep = []
    for user_id, item_id  in test_actions[["user_id", "item_id"]].itertuples(index=False):
        test_is_rep.append(int(item_id in train_val_itemsets[user_id]))
    test_actions["is_repetition"] = test_is_rep

    split_dir:Path = DATA_DIR / dataset / "split" 
    split_dir.mkdir(exist_ok=True, parents=True)

    save_mmap(train_actions, split_dir, "train")
    save_mmap(val_actions, split_dir, "val")
    save_mmap(all_val_actions, split_dir, "val_all")
    save_mmap(test_actions, split_dir, "test")

def LOO_split(txt_filepath, dataset_name:str, rand_seed = 42, n_val_users = 10000):
    '''
        TODO fix path issues
        txt_filepath is a Pathlib
        needs the all_core5.txt file to exsts
        seed is used for randomness used to select the validation users (for which the training uses n-2 length sequence)
        loads from the core k txt file of userid item id
    '''
    user_items = defaultdict(list) 
    for line in open(txt_filepath):
        user, item = line.strip().split()
        user = int(user)
        item = int(item)
        user_items[user].append(item)

    train = []
    val = []
    test = []
    all_users = sorted(list(user_items.keys()))
    rng = default_rng(rand_seed) # TODO config file for randomness for validation
    val_users = rng.choice(all_users, n_val_users, replace=False)

    for user in user_items:
        is_val_user = user in val_users 
        sequence = user_items[user]
        timestamps = range(1, len(sequence) + 1)
        for i in range(len(sequence)):
            item = sequence[i]
            timestamp = timestamps[i]
            triplet = (user, item, timestamp)
            if (i < len(sequence) - 2) or ((not is_val_user) and i == len(sequence) -2):
                train.append(triplet) # the train df will have all seqeuences of length n-1, except for  N_VAL_USERS of users will have length n-2, as the n-1th element is saved for validation
            elif ((is_val_user) and i == len(sequence) -2):
                val.append(triplet)
            else:
                assert(i == len(sequence) -1)
                test.append(triplet)
    train = pd.DataFrame(train, columns=['user_id', 'item_id', 'timestamp']) 
    test = pd.DataFrame(test, columns=['user_id', 'item_id', 'timestamp'])
    val = pd.DataFrame(val, columns=['user_id', 'item_id', 'timestamp'])
    
    save_path = processed_data_root() / dataset_name / "split"
    save_path.mkdir(exist_ok=True, parents=True)    
    save_mmap(train, save_path, "train")
    save_mmap(val, save_path, "val")
    save_mmap(test, save_path, "test")