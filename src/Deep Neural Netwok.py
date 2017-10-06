import numpy as np
from enum import Enum


class Actvitaion_Function(Enum):
    SIGMOID = 1
    ReLU = 2
    TANH = 3
    LReLU = 4


class Deep_Neural_Network:

    def __init__(self, layers_dims=None, factor=0.01, true_labels=None):
        # number of layers in the network
        self._parameters = {}
        if layers_dims is not None:
            self._depth = len(layers_dims)
            for l in range(1, self._depth):
                # initialize weights with random values
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
                # initialize biases with zeros
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

                assert (self._parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (self._parameters['b' + str(l)].shape == (layers_dims[l], 1))

    @property
    def depth(self):
        return self._depth

    @property
    def parameters(self):
        return self._parameters


def main():
    n_layer_nn = Deep_Neural_Network([5, 4, 3])
    for l in range(1, n_layer_nn.depth):
        W_k = 'W' + str(l)
        b_k = 'b' + str(l)
        print(n_layer_nn.parameters[W_k])
        print(n_layer_nn.parameters[b_k])

if __name__ == '__main__':
    main()