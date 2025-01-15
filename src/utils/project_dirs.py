from pathlib import Path

def project_root():
    current_dir = Path(__file__).absolute().parent.parent.parent
    return current_dir

def data_root():
    res = project_root()/"data"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_root():
    res = data_root() / "raw"
    res.mkdir(parents=True, exist_ok=True)
    return res

def raw_data_ds(dataset):
    res = raw_data_root()
    res = res / dataset
    res.mkdir(parents=True, exist_ok = True)
    return res

def uid_to_group_dir(dataset:str):
    res = processed_data_root() / dataset / "uid_to_group"
    res.mkdir(parents=True, exist_ok=True)
    return res

def processed_data_root() -> Path:
    res = data_root() / "processed"
    res.mkdir(parents=True, exist_ok=True)
    return res

def dataset_splits_dir(dataset):
    res: Path = processed_data_root() / dataset / "split"
    res.mkdir(exist_ok=True, parents=True)
    return res
 
def output_root():
    res =  project_root() / "output" 
    res.mkdir(parents=True, exist_ok=True)
    return res

def hparamdf_root():
    res = output_root() / "hparamdf"
    res.mkdir(parents=True, exist_ok=True)
    return res

def dataset_hparamdf_dir(dataset):
    res = hparamdf_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def dataset_group_hparamdf_dir(dataset, group):
    res = hparamdf_root() / dataset / group
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_checkpoints_root():
    res =  output_root() / "checkpoints"
    res.mkdir(parents=True, exist_ok=True)
    return res

def plot_root():
    res = output_root() / "plots"
    res.mkdir(parents=True, exist_ok=True)
    return res

def plot_dataset_root(dataset:str):
    res = plot_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def plot_dataset_int(dataset:str, inter:str): #intersecting or nonint
    res = plot_dataset_root(dataset) / inter
    res.mkdir(parents=True, exist_ok=True)
    return res


def get_irm_root():
    res =  output_root() / "irm"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_tensorboard_root():
    res =  output_root() / "tensorboard"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_model_dataset_tensorboard_dir(dataset, model):
    res = get_tensorboard_root() / dataset / model
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_dataset_checkpoints_dir(dataset):
    res = get_checkpoints_root() / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_datasetgroup_checkpoints_dir(dataset:str, group:str):
    res = get_dataset_checkpoints_dir(dataset) / group
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_dataset_group_metric_irmdir(dataset:str, group:str, metric:str):
    '''
        metric:ndcg20, recall20
    '''
    res = get_irm_root() / dataset / group / metric
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_recs_dir(dataset):
    res = output_root() / "recs" / dataset
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_recs_dir_besthyp(dataset):
    res = get_recs_dir(dataset) / "best_hyp"
    res.mkdir(parents=True, exist_ok=True)
    return res

def get_src_dir():
    return project_root() / "src" 

def get_improvements_dir():
    return get_src_dir() / "improvements"

def get_analysis_dir():
    return get_src_dir() / "analysis" 

def get_training_scripts_dir():
    return get_src_dir() / "training"

def get_eval_scripts_dir():
    return get_src_dir() / "eval"


if __name__ == "__main__":
    print(get_checkpoints_root())

