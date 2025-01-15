from pathlib import Path
import pandas as pd
from train_test_split import core5_sort_LOOsplitsave

DATA_DIR = Path("__file__").absolute().parent / "data"
DATASET_PATH = DATA_DIR / "raw/retailrocket/events.csv"

N_VAL_USERS = 10000
VAL_RAND_SEED = 42

    
def main():
    data = pd.read_csv(DATASET_PATH)
    data_views = data[data.event == "view"]
    data_views = data_views.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    data_views["timestamp"] = data_views.timestamp / 1000

    core5_sort_LOOsplitsave(df = data_views[["user_id", "item_id", "timestamp"]],
                            dataset_name = "retailrocket_views",
                            rand_seed = VAL_RAND_SEED,
                            n_val_users=N_VAL_USERS
                            )
if __name__ == "__main__":
    main()