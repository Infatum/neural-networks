from enum import Enum


class Actvitaion_Function(Enum):
    SIGMOID = 1
    ReLU = 2
    TANH = 3
    LReLU = 4


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