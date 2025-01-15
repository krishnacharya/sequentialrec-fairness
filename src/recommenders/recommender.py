class Recommender(object):
    def __init__(self) -> None:
        pass

    def train(self, train_actions, val_actions, all_val_actions, tensorboard_dir=None) -> None:
        pass

    def recommend(self, user_ids, top_k): 
        pass