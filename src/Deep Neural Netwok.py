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
        self._activation_cache = {}
        L = len(layers_dims)
        self._depth = L
        if layers_dims is not None:
            for l in range(1, L):
                # initialize weights with random values for each hidden layer
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
                # initialize biases with zeros for each hidden layer
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                assert (self._parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (self._parameters['b' + str(l)].shape == (layers_dims[l], 1))
            # init the output layer
            self._parameters['W' + str(L)] = np.random.randn(layers_dims[L - 1], layers_dims[L - 2])
            self._parameters['b' + str(L)] = np.zeros((layers_dims[L - 1], 1))

    @property
    def depth(self):
        return self._depth

    @property
    def parameters(self):
        return self._parameters

    def linear_forward(self, A, current_layer_index):
        """
        Linear part of a layer's forward propagation
        :param A: -- activations from previous layer (or input data): (size of previous layer, number of examples)
        :param current_layer_index:
        :return:
        """
        W = self._parameters['W' + str(current_layer_index)]
        b = self._parameters['b' + str(current_layer_index)]
        print('W.shape = ', W.shape)
        Z = np.dot(W, A) + b
        assert (Z.shape == (W.shape[0], A.shape[1]))
        return Z

    def

def main():
    n_layer_nn = Deep_Neural_Network([5, 4, 3])
    for l in range(1, n_layer_nn.depth):
        W_k = 'W' + str(l)
        b_k = 'b' + str(l)

    A = np.random.randn(4, 5)
    Z = n_layer_nn.linear_forward(A, 3)
    print('Z.shape = ', Z.shape)

if __name__ == '__main__':
    main()