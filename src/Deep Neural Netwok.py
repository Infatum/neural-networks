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
            self._parameters['W' + str(L)] = np.random.randn(layers_dims[L - 1], layers_dims[L - 2]) * factor
            self._parameters['b' + str(L)] = np.zeros((layers_dims[L - 1], 1))

    @property
    def depth(self):
        return self._depth

    @property
    def parameters(self):
        return self._parameters

    def __linear_forward(self, A, current_layer_index):
        """
        Linear part of a layer's forward propagation
        :param A: -- activations from previous layer (or input data): (size of previous layer, number of examples)
        :param current_layer_index:
        :return:
        """
        W = self._parameters['W' + str(current_layer_index)]
        b = self._parameters['b' + str(current_layer_index)]
        Z = np.dot(W, A) + b
        assert (Z.shape == (W.shape[0], A.shape[1]))
        return Z

    def ReLU(self, z):
        return z * (z > 0)

    def LRelU(self, z):
        """
        Leaky Rectified Linear Unit function
        :param z: -- linear function value(corresponds to WA + b)
        :return:
        """
        return z * (z > np.dot(0.001, z))

    def tanh(self, z):
        return np.tanh(z)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def activation(self, previous_activation, layer_indx, activation_type):
        """
        Forward activation step
        :param previous_activation:
        :param layer_indx:
        :param activation_type:
        :return:
        """
        Z = self.__linear_forward(previous_activation, layer_indx)
        if activation_type == Actvitaion_Function.SIGMOID:
            A = self.sigmoid(Z)
        elif activation_type == Actvitaion_Function.ReLU:
            A = self.ReLU(Z)
        elif activation_type == Actvitaion_Function.LReLU:
            A = self.LRelU(Z)
        elif activation_type == Actvitaion_Function.TANH:
            A = self.tanh(Z)
        else:
            raise ValueError('Provide a valid activation function type: either ReLU, LReLU, sigmoid or tanh')
        self._activation_cache[layer_indx] = A
        return A


def main():
    n_layer_nn = Deep_Neural_Network([5, 4, 1])
    for l in range(1, n_layer_nn.depth):
        W_k = 'W' + str(l)
        b_k = 'b' + str(l)

    X = np.random.randn(5, 40)
    A_prev = X
    for l_i in range(1, n_layer_nn.depth):
        post_activation = n_layer_nn.activation(A_prev, l_i, Actvitaion_Function.ReLU)
        print('Post activation = {0}'.format(post_activation))
        A_prev = post_activation
    post_activation = n_layer_nn.activation(A_prev, n_layer_nn.depth, Actvitaion_Function.SIGMOID)
    print('And the network output: ', post_activation)


if __name__ == '__main__':
    main()