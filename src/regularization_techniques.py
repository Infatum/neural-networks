from enum import Enum


class Regularization(Enum):
    not_required = 0
    L1 = 1
    L2 = 2
    droput = 3
    data_augmentation = 4
    early_stopping = 5
    orthogonalization = 6
    normalizing_inputs = 7