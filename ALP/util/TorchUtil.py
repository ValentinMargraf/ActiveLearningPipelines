from ALP.util.pytorch_tabnet.callbacks import Callback
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from ALP.util.transformer import TransformerModel


class TimeLimitCallback(Callback):
    """TimeLimitCallback

    This class is used for constraining TabNet classifier, such that it stops training after a certain time limit.

    Args:
        time_limit: int

    Attributes:
        time_limit: int
        start_time: float
    """
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.start_time is None:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.time_limit:
            print(f"Stopping training as the time limit of {self.time_limit} seconds has been reached.")
            return True  # This will stop training

class TabPFNEmbedder(nn.Module):
    """TabPFNEmbedder

    This class is used to create a TabPFNEmbedder model. The model is used to encode the input data into a
    lower-dimensional space.

    Args:
        X_train: numpy.ndarray
        y_train: numpy.ndarray

    Attributes:
        clf: object
        num_samples: int
        encoder: object
        fc1: object
        drop: object
        fc2: object
    """
    def __init__(self, X_train, y_train):
        super().__init__()
        self.clf = None
        self.num_samples = None
        self.instantiate_tabpfn(X_train, y_train)
        self.encoder = self.clf
        self.fc1 = nn.Linear(16384, 256)
        self.drop = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x, encode=False):
        if encode:
            x = self.encoder.predict_embeds(x)
            return torch.Tensor.numpy(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def instantiate_tabpfn(self, X_train, y_train):
        from tabpfn import TabPFNClassifier
        self.num_samples = X_train.shape[0]
        self.clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
        model = self.clf.model[2]
        ENCODER = model.encoder
        N_OUT = model.n_out
        NINP = model.ninp
        NHEAD = 4
        NHID = model.nhid
        NLAYERS = model.transformer_encoder.num_layers
        Y_ENCODER = model.y_encoder
        tf = TransformerModel(ENCODER, N_OUT, NINP, NHEAD, NHID, NLAYERS, y_encoder=Y_ENCODER)
        tf.transformer_encoder = model.transformer_encoder
        tf.decoder = model.decoder
        self.clf.model = ("inf", "inf", tf)
        self.clf.fit(X_train, y_train)
        self.clf.no_grad = True