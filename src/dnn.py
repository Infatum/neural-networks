from base_neural_network import Base_Neural_Network
import numpy as np
from dnn_types import Initialization_Type
from dnn_types import  Actvitaion_Function
from regularization_techniques import Regularization


class Deep_Neural_Network(Base_Neural_Network):

    def __init__(self, layers_dims, initialization_type=Initialization_Type.He, factor=0.075):
        super(Deep_Neural_Network, self).__init__(layers_dims, initialization_type, factor)
        # self._initialize_network(layers_dims, initialization_type, factor)

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

    def backward_propagation(self, Y, regularization=Regularization.L2, lambd=0.1):
        grads = None
        if regularization == Regularization.not_required:
            grads = super(Deep_Neural_Network, self).backward_propagation(Y)
        elif regularization == Regularization.L2:
            grads = self.__back_prop_with_regularization(Y, lambd)
        elif regularization == Regularization.L1:
            raise NotImplemented('Not implemented yet')
        elif regularization == Regularization.droput:
            # todo: implement droput
            raise NotImplemented('Not implemented yet')
        else:
            raise ValueError('Back prop doesn''t use any other regularization techniques then: L1, L2, Dropout')

    def __back_prop_with_regularization(self, Y, lamdb):
        iters, m = self._depth - 1, self._features.shape[1]
        A_L = self._activation_cache[self._depth - 1]
        dZ_l = A_L - Y
        grads = {}
        dA_l = np.dot(self.parameters['W' + str(self._depth)], dZ_l)

        for l in reversed(range(iters)):
            A_prev, A = self._activation_cache[l - 1], self._activation_cache[l - 1]
            W_l, b_l = self._parameters['W' + str(l)], self._parameters['b' + str(l)]
            grads['dW' + str(l)] = 1./m * np.dot(dZ_l, A_prev.T) + np.dot(lamdb / m, W_l)
            grads['db' + str(l)] = 1./m * np.sum(dZ_l, axis=1, keepdims=True)
            dA_l = np.dot(self._parameters['W' + str(l - 1)], dZ_l)
            dZ_l = np.multiply(dA_l, np.int64(A > 0))

        return grads


def main():
    dnn_model = Deep_Neural_Network((4, 5, 3, 1), initialization_type=Initialization_Type.He)


if __name__ == '__main__':
    main()