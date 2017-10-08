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
        :returns
            A: -- post-activation value
            Z: -- linear cache(linear function value)
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
        return A, Z

    def forward_propagation(self, X):
        """
        Forward propagation step for overall Neural Net structure
        :param X: -- input data(i.e. input matrix)
        :returns
            cache: -- activation cache for each layer
            Z: -- linear cache for each layer
        """
        cache = []
        A = X

        for l in range(1, self._depth):
            A_prev = A
            A, Z = self.activation(A_prev, l, Actvitaion_Function.ReLU)
            cache.append(A)

        net_output = self.activation(A, self._depth, Actvitaion_Function.SIGMOID)
        cache.append(net_output)
        return Z, cache

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

    def __derivation(self, dZ, layer_index, activation_cache):
        """
        Linear portion for a backward propagation
        :param dZ: -- Gradient of the cost with respect to the linear output (of current layer l)
        :param activation_cache: -- list of cached activation value for each layer with L-1 indexing
        :returns
            dA_prev: -- Gradient of the cost with respect to the activation (of the previous layer l-1)
            dW: -- Gradient of the cost with respect to W (current layer l)
            db: -- Gradient of the cost with respect to b (current layer l)
        """
        # activation indexing
        l = layer_index - 1
        A, A_prev = activation_cache[l], activation_cache[l - 1]
        W, b = self._parameters['W' + str(layer_index)], self._parameters['b' + str(layer_index)]
        # amount of neurons in the previous layer
        m = A_prev.shape[1]

        dW = np.dot(1 / m, np.dot(dZ, A_prev.T))
        db = np.dot(1 / m, np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def __sigmoid_gradient(self, dA, A):
        """
        Calculates sigmoid function gradient
        :param dA: -- post-activation gradient
        :param A: -- linear activation function cache 'Z'
        :returns
            dZ: -- gradient of the cost with respect to Z
        """
        s = 1 / (1 + np.exp(-A))
        dZ = dA * s * (1 - s)
        return dZ

    def __ReLU_gradient(self, dA, A):
        """
        Calculates ReLU function gradient
        :param dA: -- post-activation gradient
        :param A: -- linear activation function cache 'Z'
        :returns
            dZ: -- gradient of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True)
        dZ[A <= 0] = 0
        return dZ
    # todo: realize tanh and LRelU gradients

    def gradient_descent(self, dA, linear_cache, activation_cache, activation_type):
        """
        Gradient descent step for a backward propagation step
        :param dA:
        :param linear_cache:
        :param activation_cache:
        :param activation_type:
        :return:
        """
        if activation_type == Actvitaion_Function.SIGMOID:
            dZ = self.__sigmoid_gradient(dA, activation_cache)
        elif activation_type == Actvitaion_Function.ReLU:
            dZ = self.__ReLU_gradient(dA, activation_cache)



def main():
    n_layer_nn = Deep_Neural_Network([5, 4, 2, 1])
    X = np.random.randn(5, 40)
    A_prev = X
    activations = n_layer_nn.forward_propagation(X)
    l = 1
    for a in activations:
        print('Activation of the {0}th layer = {1}'.format(l, a))
        l += 1
    dZ = np.dot()


if __name__ == '__main__':
    main()