import torch

class SequentialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    