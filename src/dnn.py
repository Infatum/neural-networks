import math
import numpy as np
from dnn_types import Data_Type
from  dnn_types import Learning
from dnn_types import Initialization_Type
from dnn_types import Actvitaion_Function
from base_neural_network import Base_Neural_Network
from regularization_techniques import Regularization

# todo: this -> add auto-doc to class methods and constructor
class Deep_Neural_Network(Base_Neural_Network):

    def __init__(self, layers_dims, mini_batch_size=64, learning_type=Learning.Supervised,
                 regularization_type=Regularization.not_required, keep_probabilities=None,
                 initialization_type=Initialization_Type.He, factor=0.075):
        # init DNN weights and biases
        super(Deep_Neural_Network, self).__init__(layers_dims, initialization_type, factor)
        self._regularization = regularization_type
        self._drop_out_mask = {}
        self._mini_batch_size = mini_batch_size
        self._learning_type = learning_type

        if regularization_type == Regularization.droput:
            if keep_probabilities is not None:
                try:
                    assert (len(keep_probabilities) == self._depth - 1)
                except AssertionError:
                    raise ValueError('Length of the drop_out list should be applied only to hidden layers, '
                                     'output layer should be unchanged')
                self._keep_probabilities = keep_probabilities
            else:
                raise ValueError('Provide a list with probability of drop_out for each layer with length DNN.depth - 1')

    def _initialize_network(self, layers_dims, init_type, factor=0.075):
        print('Deep Neural Network init')
        if init_type == Initialization_Type.random:
            super(Deep_Neural_Network, self)._initialize_network(layers_dims, factor)
        elif init_type == Initialization_Type.He:
            self.__he_init(layers_dims)
        elif init_type == Initialization_Type.ReLU:
            self.__ReLU_init(layers_dims)
        elif init_type == Initialization_Type.Xavier:
            self.__xavier_init(layers_dims)
        else:
            raise NotImplemented('Other init types haven'' been implemented')

    def __xavier_init(self, layers_dims):
        """
        initialize weights with improved Xavier initialization for each hidden layer and biases with zeros

        :param layers_dims: -- dimensions structure of the layers for DNN
        """
        if layers_dims is not None:
            for l in range(1, self._depth + 1):
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
                    layers_dims[l - 1])
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        else:
            raise ReferenceError('Provide a list of DNN structure, '
                                 'where each element should describe amount of neurons and it''s index - layer index')

    def __ReLU_init(self, layers_dims):
        if layers_dims is not None:
            for l in range(1, self._depth + 1):
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                        2 / (layers_dims[l - 1] + layers_dims[l]))
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        else:
            raise ReferenceError('Provide a list of DNN structure, '
                                     'where each element should describe amount of neurons and it''s index - layer index')

    def __he_init(self, layers_dims):
        np.random.seed(3)
        if layers_dims is not None:
            for l in range(1, self._depth + 1):
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
                        2 / (layers_dims[l - 1]))
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        else:
            raise ReferenceError('Provide a list of DNN structure, '
                                     'where each element should describe amount of neurons and it''s index - layer index')

    def _prepare_mini_batches(self, features, labels=None, learning_type=Learning.Supervised):
        mini_batches = []

        if self._learning_type == Learning.Supervised:
            m = features.shape[1]
            permutation = list(np.random.permutation(m))
            shuffled_f, shuffled_l = features[:, permutation], labels[:, permutation]
            number_of_complete_minibatches = math.floor(m / self._mini_batch_size)

            for i in range(0, number_of_complete_minibatches):
                mini_batch_X = shuffled_f[:, i * self._mini_batch_size: (i + 1) * self._mini_batch_size]
                mini_batch_Y = shuffled_l[:, i * self._mini_batch_size: (i + 1) * self._mini_batch_size]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            if m % self._mini_batch_size != 0:
                mini_batch_X = shuffled_f[:, number_of_complete_minibatches * self._mini_batch_size:]
                mini_batch_Y = shuffled_l[:, number_of_complete_minibatches * self._mini_batch_size:]
                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

        elif self._learning_type == Learning.Unsupervised:
            m = features.shape[0]
            permutation = list(np.random.permutation(m))
            shuffled_f = features[:, permutation]
            number_of_complete_minibatches = math.floor(m / self._mini_batch_size)

            for i in range(0, number_of_complete_minibatches):
                mini_batch = shuffled_f[:, i * self._mini_batch_size: (i + 1) * self._mini_batch_size]
                mini_batches.append(mini_batch)

            if m % self._mini_batch_size != 0:
                mini_batch = shuffled_f[:, number_of_complete_minibatches * self._mini_batch_size:]
                mini_batches.append(mini_batch)
        else:
            raise NotImplementedError('Haven''t implemented mini batch preparation for reinforcement learning yet')
        return mini_batches

    def forward_propagation(self, X):
        if self._regularization == Regularization.droput:
            self._features = X
            A = X

            for l in range(1, self._depth):
                A_previous = A
                A, Z = super(Deep_Neural_Network, self).activation(A_previous, l, Actvitaion_Function.ReLU)
                self._linear_cache.append(Z)
                D = self.__forward_drop_out(A, l - 1)
                A = np.multiply(A, D)
                A /= self._keep_probabilities[l - 1]
                self._activation_cache.append(A)
                self._drop_out_mask['D' + str(l)] = D
                # kept neurons after using drop out
            net_output, Z = self.activation(A, self._depth, Actvitaion_Function.SIGMOID)
            self._activation_cache.append(net_output, Z)
            assert (net_output.shape == (1, X.shape[1]))
            return net_output
        else:
            return super(Deep_Neural_Network, self).forward_propagation(X)

    def __forward_drop_out(self, current_activation, activation_index):
        droped_out = np.random.randn(current_activation.shape[0], current_activation.shape[1])
        droped_out = droped_out < self._keep_probabilities[activation_index]
        return droped_out

    def compute_cost(self, net_output, Y, lambd=0.5):
        if self._regularization == Regularization.L1:
            raise NotImplementedError('Currently not implemented type of regularization')
        elif self._regularization == Regularization.L2:
            cross_entropy_cost = super(Deep_Neural_Network, self).compute_cost(net_output, Y)
            return self.__compute_cost_with_regularization(cross_entropy_cost, lambd)
        else:
            return super(Deep_Neural_Network, self).compute_cost(net_output, Y)

    def __compute_cost_with_regularization(self, cross_entropy_cost, lambd):
        """

        :param net_output:
        :param Y:
        :param layer_index:
        :param lambd:
        :return:
        """
        W, m = self._features, self._depth + 1
        summ = np.sum(np.square(W))

        for l in range(1, m):
            W = self._parameters['W' + str(l)]
            summ += np.sum(np.square(W))

        cost = (summ * lambd / (2 * m)) + cross_entropy_cost
        return cost

    def backward_propagation(self, Y, lambd=0.1):
        grads = None
        if self._regularization == Regularization.L2:
            grads = self.__back_prop_with_regularization(Y, lambd)
        elif self._regularization == Regularization.L1:
            raise NotImplemented('Not implemented yet')
        elif self._regularization == Regularization.droput:
            grads = self.__backward_drop_out()
        else:
            grads = super(Deep_Neural_Network, self).backward_propagation(Y)

        return grads

    def __backward_drop_out(self, Y):
        m = self._features.shape[1]
        grads = {}
        dZ = self._activation_cache[self._depth - 1] - Y
        A_prev = self._activation_cache[self._depth - 2]
        grads['dW' + str(self._depth)] = 1./m * np.dot(dZ, A_prev.T)
        grads['db' + str(self._depth)] = 1./m * np.sum(dZ, axis=1, keepdims=True)
        W, b = self._parameters['W' + str(self._depth)], self._parameters['b' + str(self._depth)]

        for l in reversed(range(self._depth - 1)):
            A = self._activation_cache[l - 1]
            dA = np.multiply(W.T, dZ)
            dA = np.multiply(dA, self._drop_out_mask['D' + str(l)])
            dA /= self._keep_probabilities[l - 1]
            dZ = np.multiply(dA, np.int64(A > 0))

        return grads

    def __back_prop_with_regularization(self, Y, lamdb):
        iters, m = self._depth - 1, self._features.shape[1]
        A = self._activation_cache[self._depth - 1]
        dZ = A - Y
        grads = {}

        for l in reversed(range(iters)):
            A_prev, A = self._activation_cache[l - 1], self._activation_cache[l]
            W, b = self._parameters['W' + str(l)], self._parameters['b' + str(l)]
            grads['dW' + str(l)] = 1./m * np.dot(dZ, A_prev.T) + np.dot(lamdb / m, W)
            grads['db' + str(l)] = 1./m * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(W, dZ)
            dZ = np.multiply(dA, np.int64(A > 0))

        return grads


def main():
    dnn_model = Deep_Neural_Network((4, 5, 3, 1), initialization_type=Initialization_Type.He)


if __name__ == '__main__':
    main()