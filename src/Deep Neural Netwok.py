import numpy as np
from enum import Enum


class Actvitaion_Function(Enum):
    SIGMOID = 1
    ReLU = 2
    TANH = 3
    LReLU = 4


class Deep_Neural_Network:

    def __init__(self, layers_dims=None, factor=0.01, true_labels=None):
        """
        Initialize the N-Layer Neural Net structure
        :param layers_dims: -- layers dimensions, where layers_dims[0] is a feature vector
        :param factor: -- small-scale factor value to reduce initial weights and biases values
        :param true_labels: -- labeled with correct answers data-set for a supervised learning training
        """
        # number of layers in the network
        self._parameters = {}
        self._activation_cache = {}
        L = len(layers_dims)
        self._depth = L - 1
        if layers_dims is not None:
            for l in range(1, L):
                # initialize weights with random values for each hidden layer
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
                # initialize biases with zeros for each hidden layer
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                assert (self._parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (self._parameters['b' + str(l)].shape == (layers_dims[l], 1))

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
        :returns Z: -- linear function value(WA[l-1] + b[l])
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
        Activation of the i-th layer
        :param previous_activation: -- activations from previous layer (or input data): (size of previous layer, number of examples)
        :param layer_indx: -- layer index
        :param activation_type: -- activation function type
        :returns A: -- post-activation value
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
        #self._activation_cache[layer_indx] = A
        return A

    def forward_propagation(self, X):
        """
        Forward propagation step for overall Neural Net structure
        :param X: -- input data(i.e. input matrix)
        :returns cache: -- activation cache for each layer
        """
        cache = []
        A = X

        for l in range(1, self._depth):
            A_prev = A
            A = self.activation(A_prev, l, Actvitaion_Function.ReLU)
            cache.append(A)

        net_output = self.activation(A, self._depth, Actvitaion_Function.SIGMOID)
        cache.append(net_output)
        return cache

    def compute_cost(self, net_output, Y):
        """
        Computes the cost over all training set
        :param net_output:
        :param Y: -- true "label" vector
        :returns cost: -- cross-entropy cost(a mean value for the cross-entropy loss over all training examples)
        """
        # m = amount of training examples
        m = Y.shape[1]
        cost = -np.dot(1 / m, np.sum(np.dot(net_output).T) + np.dot(1 - Y, np.log(1 - net_output).T))
        cost = np.squeeze(cost)
        return cost


def main():
    n_layer_nn = Deep_Neural_Network([5, 4, 1])
    X = np.random.randn(5, 40)
    A_prev = X
    activations = n_layer_nn.forward_propagation(X)
    l = 1
    for a in activations:
        print('Activation of the {0}th layer = {1}'.format(l, a))
        l += 1


if __name__ == '__main__':
    main()