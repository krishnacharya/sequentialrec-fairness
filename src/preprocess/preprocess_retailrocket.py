from pathlib import Path
import pandas as pd
from numpy.random import default_rng
from train_test_split import split_data
import zipfile

DATA_DIR = Path("__file__").absolute().parent / "data"
ARCHIVE_PATH = DATA_DIR / "raw/retailrocket/archive.zip"


def process_data(data: pd.DataFrame, dataset_name, n_val_users=1024):
    data = data[["user_id", "item_id", "timestamp"]]
    data = data.sort_values("timestamp")
    output_dir = Path(DATA_DIR/"processed"/dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "all.csv", index=False)
    split_data(dataset_name, n_val_users=n_val_users)
    



def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError("Please download Retailrocket Dataset from https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset and put it into the 'retailrocket' forlder")

    archive = zipfile.ZipFile(ARCHIVE_PATH) 
    with  archive.open('events.csv') as input:
        data = pd.read_csv(input)
    data_views = data[data.event == "view"]
    data_views = data_views.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    data_views["timestamp"] = data_views.timestamp // 1000
    process_data(data_views, "retailrocket_views")



if __name__ == "__main__":
    main()