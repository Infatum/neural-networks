import math
import numpy as np
from dnn_types import Data_Type
from  dnn_types import Optimization_Type
from dnn_types import Initialization_Type
from dnn_types import Actvitaion_Function
from base_neural_network import Base_Neural_Network
from regularization_techniques import Regularization


# todo: this -> add auto-doc to class methods and constructor
# todo: add batch normalization
class Deep_Neural_Network(Base_Neural_Network):

    def __init__(self, layers_dims, layers_activations, mode, mini_batch_size=64, batch_norm=True,
                 optimizer=Optimization_Type.Adam,
                 regularization_type=Regularization.not_required, keep_probabilities=None,
                 initialization_type=Initialization_Type.He, factor=0.075):
        # init DNN weights and biases
        super(Deep_Neural_Network, self).__init__(layers_dims, layers_activations, mode, initialization_type, factor)
        self._batch_norm = batch_norm
        self._regularization = regularization_type
        self._drop_out_mask = {}
        self._mini_batch_size = mini_batch_size
        self._optimizer = optimizer

        if regularization_type == Regularization.droput:
            if keep_probabilities is not None:
                try:
                    assert (len(keep_probabilities) == self._depth - 1)
                except AssertionError:
                    raise ValueError('Drop out list should be applied only to hidden layers, and'
                                     ' have the dimension of layers_quantity - 1, the output layer should be unchanged')
                self._keep_probabilities = keep_probabilities
            else:
                raise ValueError('Provide a list with probability of drop_out for each layer with length DNN.depth - 1')

    def _initialize_network(self, layers_dims, init_type, factor=0.075):
        print('Deep Neural Network init')
        if init_type == Initialization_Type.random:
            super(Deep_Neural_Network, self)._initialize_network(layers_dims, factor)
        else:
            for l in range(1, self._depth + 1):
                if self._batch_norm:
                    self._init_batch_norm(l, layers_dims[l], layers_dims[l - 1])
                if init_type ==  Initialization_Type.He:
                    self.__he_init(l, layers_dims[l], layers_dims[l - 1])
                elif init_type == Initialization_Type.Xavier:
                    self.__xavier_init(l, layers_dims[l], layers_dims[l - 1])
                elif init_type == Initialization_Type.ReLU:
                    self.__ReLU_init(l, layers_dims[l], layers_dims[l - 1])
                else:
                    raise NotImplementedError('Haven''t implemented yet')

    def _init_batch_norm(self, layer, current_dim, previous_dim):
        self._parameters['Gamma' + str(layer)] = np.ones((current_dim, previous_dim))
        self._parameters['Beta' + str(layer)] = np.zeros((current_dim, previous_dim))

    def __xavier_init(self, layer, current_dim, previous_dim):
        """
        initialize weights with improved Xavier initialization for each hidden layer and biases with zeros

        :param layers_dims: -- dimensions structure of the layers for DNN
        """
        l = layer
        self._parameters['W' + str(l)] = np.random.randn(current_dim, previous_dim) / np.sqrt(previous_dim)
        self._parameters['b' + str(l)] = np.zeros((current_dim, 1))

    def __ReLU_init(self, layer, current_dim, previous_dim):
        l = layer
        self._parameters['W' + str(l)] = np.random.randn(current_dim, previous_dim) * np.sqrt(
                        2 / (previous_dim + current_dim))
        self._parameters['b' + str(l)] = np.zeros((current_dim, 1))

    def __he_init(self, layer, current_dim, previous_dim):
        l =  layer
        self._parameters['W' + str(l)] = np.random.randn(current_dim, previous_dim) * np.sqrt(2 / (previous_dim))
        self._parameters['b' + str(l)] = np.zeros((current_dim, 1))

    def _prepare_mini_batches(self, features, labels):
        mini_batches = []
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
        return mini_batches

    # todo: add batch norm functionality
    def forward_propagation(self, X, epsilon=0.00001):
        if self._batch_norm:
            X = self.normalize_inputs(X)
        self._features = X
        A = X

        for l in range(1, self._depth):
            A_previous = A
            A, Z = super(Deep_Neural_Network, self).activation(A_previous, l, self._layers_activations[l])
            if self._batch_norm:
                Z = self._forward_prop_with_batch_norm(l, Z, epsilon)
            self._linear_cache.append(Z)
            if self._regularization == Regularization.droput:
                A = self._drop_out(A, l - 1)
            self._activation_cache.append(A)
            # kept neurons after using drop out
        net_output, Z = self.activation(A, self._depth, self._layers_activations[self._depth])
        self._activation_cache.append(net_output, Z)
        assert (net_output.shape == (1, X.shape[1]))
        return net_output

    def normalize_inputs(self, X):
        mean = np.mean(X)
        variance = (1. / X.size[1]) * np.sum(X ** 2)
        X /= variance
        return X
    def __drop_out(self, current_activation, activation_index):
        droped_out = np.random.randn(current_activation.shape[0], current_activation.shape[1])
        droped_out = droped_out < self._keep_probabilities[activation_index]
        current_activation = np.multiply(current_activation, droped_out)
        current_activation /= self._keep_probabilities[activation_index]
        self._drop_out_mask['D' + str(activation_index + 1)] = droped_out
        return current_activation

    def _forward_prop_with_batch_norm(self, layer, Z, epsilon):
        m = Z.shape[1]
        mean = np.mean(Z)
        varience = (1./ m) * np.sum(np.square(Z - mean))
        Z_norm = (Z - mean) / np.sqrt(varience + epsilon)
        Z_tilda = np.dot(self._parameters['Gamma' + str(layer)], Z_norm) + self._parameters['Beta' + str(layer)]
        return Z_tilda

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
        grads = super(Deep_Neural_Network, self).backward_propagation(Y)
        m = self._features[1]
        adam, momentum, v, s = False, False, None, None
        if self._optimizer == Optimization_Type.Adam:
            adam = True
            v = self.__initialize_velocity()
        elif self._optimizer == Optimization_Type.Momentum:
            momentum = True
            v, s = self.__initialize_adam()

        if self.__class__.__name__ == 'Deep_Neural_Network':
            dA_prev = grads['dA' + str(self._depth)]
            for l in reversed(range(self._depth - 1)):
                layer = l + 1
                W, b = self._parameters['W' + str(layer)], self._parameters['b' + str(layer)]
                dA, dW, db = super(Deep_Neural_Network, self)._compute_gradients(dA_prev, layer,
                                                                                      self._layers_activations[l])
                if self._regularization == Regularization.L2:
                    # L2 Regularization back prop part
                    # todo: should I shift this to the separate function? (logic separation)
                    dW += np.dot(lambd / m, W)
                elif self._regularization == Regularization.droput:
                    # dropout backprop part
                    # todo: should I shift this to the separate function? (logic separation)
                    dA = np.multiply(dA, self._drop_out_mask['D' + str(layer)])
                    dA /= self._keep_probabilities[l]
                elif self._regularization == Regularization.not_required:
                    pass
                else:
                    raise NotImplementedError('Other types of regularization aren''t implemented yet')
                dA_prev = dA

        return grads

    def __initialize_velocity(self):
        v = {}

        for l in range(self._depth):
            dW_shape, db_shape = self._parameters['W' + str(l + 1)].shape, self._parameters['b' + str(l + 1)].shape
            v['dW' + str(l + 1)] = np.zeros((dW_shape[0], dW_shape[1]))
            v['db' + str(l + 1)] = np.zeros((db_shape[0], db_shape[1]))
        return v

    def _update_parameters_with_momentum(self, grads, layer, learning_rate, v, beta=0.9):
        """
        Computes velocity with respect to gradients for gradient descent with momentum optimization

        :param grads: -- weights and biases gradients
        :param learning_rate: --
        :param v:
        :param beta:
        :return:
        """
        W, b,  = self._parameters['W' + str(layer)], self._parameters['b' + str(layer)]
        dW, db = grads['dW' + str(layer)], grads['db' + str(layer)]
        v_dW = beta * v['dW' + str(layer)] + (1 - beta) * grads['dW' + str(layer)]
        v_db = beta * v['db' + str(layer)] + (1 - beta) * grads['db' + str(layer)]
        W, b = self._parameters['W' + str(layer)], self._parameters['b' + str(layer)]
        self._parameters['W' + str(layer)] = W - learning_rate * v_dW
        self._parameters['W' + str(layer)] = b - learning_rate * v_db

    def __initialize_adam(self):
        v, s = {}, {}

        for l in range(1, self._depth + 1):
            v['dW' + str(l)] = np.zeros(self._parameters['W'].shape)
            v['db' + str(l)] = np.zeros(self._parameters['b'].shape)
            s['dW' + str(l)] = np.zeros(self._parameters['W'].shape)
            s['db' + str(l)] = np.zeros(self._parameters['b'].shape)
        return v, s

    # todo: update parameters with respect to regularization and optimization
    def update_parameters(self, grads, learning_rate, lambd=0.01):
        return

    def _update_parameters_with_adam(self, grads, layer, learning_rate, v, s, t, beta1=0.9,
                                     beta2=0.999, epsilon = 1e-8):
        v_dW, v_db = v
        s_dW, s_db = s
        dW, db = grads

        v_dW, v_db = beta1 * v_dW + (1 - beta1) * dW, beta1 * v_db + (1 - beta1) * db
        v_dW_corrected, v_db_corrected = v_dW / (1 - np.power(beta1, t)), v_db / (1 - np.power(beta1, t))

        s_dW, s_db = beta2 * s_dW + (1 - beta2) * np.square(dW), beta2 * s_db + (1 - beta2) * np.square(db),
        s_dW_corrected, s_db_corrected = s_dW / (1 - np.power(beta2, t)), s_db / (1 - np.power(beta2, t))

        W, b = self._parameters['W' + str(layer)], self._parameters['b' + str(layer)]
        self._parameters['W' + str(layer)] = W - learning_rate * v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon)
        self._parameters['b' + str(layer)] = b - learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + epsilon)


def main():
    dnn_model = Deep_Neural_Network((4, 5, 3, 1), initialization_type=Initialization_Type.He)


if __name__ == '__main__':
    main()