from enum import Enum


class Actvitaion_Function(Enum):
    SIGMOID = 1
    ReLU = 2
    TANH = 3
    LReLU = 4
    SOFTMAX = 5


class Initialization_Type(Enum):
    Xavier = 1
    He = 2
    ReLU = 3
    random = 4


class Data_Type(Enum):
    Features = 1
    Labels = 2


class Learning(Enum):
    Supervised = 1
    Unsupervised = 2
    Reinforcment_Learning = 3


class Optimization_Type(Enum):
    Only_mini_batch = 1
    Momentum = 2
    RMSProp = 3
    Adam = 4


class NN_Mode(Enum):
    Binary_Classification = 1
    Multiclass_Classification = 2
    Regression = 3
