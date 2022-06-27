from enum import Enum
import torch
import torch.nn as nn


class Optimizer(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class Loss(Enum):
    XENT = "crossentropy"


class PoolingMethod(Enum):
    GAP1D = "gap"
    FLATTEN = "flatten"


class RNN(Enum):
    LSTM = "lstm"
    GRU = "gru"


class Activation(Enum):
    RELU = "relu"
    ELU = "elu"
    SIGMOID = "sigmoid"


def get_activation(activation: Activation) -> nn.Module:
    if isinstance(activation, str):
        activation = Activation(activation.lower().strip())
    if activation.value == Activation.RELU.value:
        return nn.ReLU()
    elif activation.value == Activation.SIGMOID.value:
        return nn.Sigmoid()
    elif activation.value == Activation.ELU.value:
        return nn.ELU()


def get_optimizer(name: Optimizer) -> torch.optim.Optimizer:
    if isinstance(name, str):
        name = Optimizer(name.lower().strip())
    if name.value == Optimizer.ADAM.value:
        return torch.optim.Adam
    elif name.value == Optimizer.ADAMW.value:
        return torch.optim.AdamW
    elif name.value == Optimizer.SGD.value:
        return torch.optim.SGD
    elif name.value == Optimizer.RMSPROP.value:
        return torch.optim.RMSprop
    else:
        raise NameError(f"Unrecognized optimizer name: {name}")


def get_loss_fn(name: Loss) -> nn.Module:
    if isinstance(name, str):
        name = Loss(name.lower().strip())
    if name.value == Loss.XENT.value:
        return nn.CrossEntropyLoss()
    else:
        raise NameError(f"Unrecognized loss name: {name}")