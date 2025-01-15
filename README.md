Codebase for the paper `Improving Minimax Group Fairness in Sequential Recommendation`

## Structure

The source code is in the `src` directory and the data in the `data` directory.

1. The implementation of the sequential recommender and training methods is in `src/recommenders`.
    - `src/recommenders/SASRec.py` contains the SASRec recommender, and `src/recommenders/utils/sequencer.py` contains the sequencer used for padding/truncation of each user's item sequence.
    - The distributionally robust methods and baselines in the paper are available in `src/recommenders/utils/loss.py`.
    - `src/recommenders/utils/metric_computer.py` and `src/recommenders/utils/checkpointer.py` are used for calculating metrics and checkpointing the best model, respectively.

2. `src/training/train_sasrec_dp` and `src/sagemaker/sagemaker_training.py` are used to launch jobs locally and on AWS Sagemaker.

3. Finally, the improvements compared to standard training are in `src/improvements`. The RR_improvement.ipynb and ML1m_improvement.ipynb notebooks contain the analysis of the RetailRocket and Movielens1M experiments.

## Datasets and Processing

We use Movielens1M and RetailRocket, which are popular open datasets for movies and e-commerce.

1. Download the processed ML1m from https://github.com/FeiSun/BERT4Rec/blob/master/data/ml-1m.txt and place it in `data/raw/ml1m`. Download the `events.csv` from https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset and place it in `data/raw/retailrocket`.
2. We have already preprocessed the two datasets by running `src/preprocess/preprocess_movielens1m.py` and `src/preprocess/preprocess_retailrocket.py`, respectively. The processed data is in `data/processed`. We have also added user groups by running `src/preprocess/addgroups_dsplit.py`.

## Usage

Note that the PYTHONPATH for your conda environment should be set to the base of this repository.

### Local Run

We first suggest running `python src/training/train_sasrec_dp.py`, this script has default values for the training parameters.

The key parameters we change for the experiments:
- The loss type is configurable to one of: `["joint_dro", "erm", "cb", "cb_log", "s_dro", "group_dro", "ipw", "ipw_log"]`.
- `joint-dro-alpha` is the alpha level for CVaR DRO.
- The `gdro-stepsize` is the stepsize for the ascent step in the group and streaming dro loss.
- `stream-lr` is the streaming learning rate for SDRO.
- `groups` selects which user group and size we use in the experiment:
    - `[popdsplit_balanced, popdsplit_0.2_0.6_0.2, popdsplit_0.1_0.8_0.1]` are G_pop33, G_pop2060, and G_pop1080 from the paper.
    - `[seqdsplit_balanced, seqdsplit_0.2_0.6_0.2, seqdsplit_0.1_0.8_0.1]` are G_seq33, G_seq2060, and G_seq1080 from the paper.
    - `popseq_bal`
- `subgroup-for-loss` selects which subgroup to use for loss computation for GDRO, SDRO, and IPW when users belong to both popularity and sequence length groups. `0` uses popularity-based groups, `1` uses sequence length groups.

### Launching Jobs on Sagemaker

Launching jobs on Sagemaker is straightforward via `python src/sagemaker/sagemaker_training.py <config path>` with the appropriate config path. 
The configs for all the jobs in our experiments are available in `src/sagemaker/training_configs`, for e.g. the command:

`python src/sagemaker/sagemaker_training.py 'src/sagemaker/training_configs/RR/sasrec/popseq/rr-sasrec-jdro-popseq.json'`

Launches CVaR DRO (also called jointDRO) training jobs with 10 different alpha values, on the RetailRocket dataset for users belonging to intersecting groups (popularity and sequence length-based groups).

--- 
