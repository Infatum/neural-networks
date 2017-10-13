import numpy as np
from enum import Enum
import test_cases as ts_cs


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
        self._activation_cache = []
        self._linear_cache = []
        # todo: features (cache or not - that's the question'
        self._features = None

        L = len(layers_dims)
        self._depth = L - 1

        if layers_dims is not None:
            for l in range(1, L):
                # initialize weights with random values for each hidden layer
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
            layers_dims[l - 1])
                # initialize biases with zeros for each hidden layer
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                assert (self._parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (self._parameters['b' + str(l)].shape == (layers_dims[l], 1))

    @property
    def activations(self):
        A = tuple(self._activation_cache)
        return A

    @property
    def depth(self):
        return self._depth

    @property
    def parameters(self):
        # todo: return an immutable data structure to avoid changes of the dnn parameters
        return self._parameters

    def __linear_forward(self, A, current_layer_index):
        """
        Linear part of a layer's forward propagation

        :param A: -- activations from previous layer (or input data): (size of previous layer, number of examples)
        :param current_layer_index:
        :returns Z: -- linear function value(WA[l-1] + b[l])
        """
        # todo: find out why W differs from Coursera's version
        W = self._parameters['W' + str(current_layer_index)]
        b = self._parameters['b' + str(current_layer_index)]
        Z = np.dot(W, A) + b
        assert (Z.shape == (W.shape[0], A.shape[1]))
        return Z

    def ReLU(self, z):
        """
        Rectified Linear Unit function (activation f-n)

        :param z: -- linear function value(corresponds to WA + b)
        :returns: -- post activation value
        """
        return z * (z > 0)

    def LRelU(self, z):
        """
        Leaky Rectified Linear Unit function (activation f-n)

        :param z: -- linear function value(corresponds to WA + b)
        :returns: -- post activation value
        """
        return z * (z > np.dot(0.001, z))

    def tanh(self, z):
        """
        Hyperbolic tangent function (activation f-n)

        :param z: -- linear function value(corresponds to WA + b)
        :returns: -- post activation value
        """
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
        # todo: debug incorrect linear function calculation(Z)
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
        return A, Z

    def forward_propagation(self, X):
        """
        Forward propagation step for overall Neural Net structure

        :param X: -- input data(i.e. input matrix)
        :returns
            net_output: -- activation of the last(output) layer
            Z: -- linear cache for each layer
        """
        cache = []
        A = X
        self._features = X

        for l in range(1, self._depth):
            A_prev = A
            # todo: debug activation mismatch
            A, Z = self.activation(A_prev, l, Actvitaion_Function.ReLU)
            self._activation_cache.append(A)
            self._linear_cache.append(Z)

        net_output, Z = self.activation(A, self._depth, Actvitaion_Function.SIGMOID)
        self._activation_cache.append(net_output)
        self._linear_cache.append(Z)

        assert (net_output.shape == (1, X.shape[1]))
        return net_output

    def compute_cost(self, net_output, Y):
        """
        Computes the cost over all training set

        :param net_output:
        :param Y: -- true "label" vector
        :returns cost: -- cross-entropy cost(a mean value for the cross-entropy loss over all training examples)
        """
        # m = amount of training examples
        m = Y.shape[1]
        cost = (1. / m) * (-np.dot(Y, np.log(net_output).T) - np.dot(1 - Y, np.log(1 - net_output).T))
        cost = np.squeeze(cost)
        return cost

    def __derivation(self, dZ, layer_index):
        """
        Linear portion for a backward propagation step

        :param dZ: -- Gradient of the cost with respect to the linear output (of current layer l)
        :param activation_cache: -- list of cached activation value for each layer with L-1 indexing
        :returns
            dA_prev: -- Gradient of the cost with respect to the activation (of the previous layer l-1)
            dW: -- Gradient of the cost with respect to W (current layer l)
            db: -- Gradient of the cost with respect to b (current layer l)
        """
        A = self._activation_cache[layer_index - 1]
        # get previous layer activation values if layer isn't first, either assign prev. activ. to the feature vector
        # todo: find out what should be an A_prev value for the first layer
        A_prev = self._activation_cache[layer_index - 2] if layer_index > 1 else self._features
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

    def __sigmoid_gradient(self, dA, Z):
        """
        Calculates sigmoid function gradient

        :param dA: -- post-activation gradient
        :param A: -- linear activation function cache 'Z'
        :returns
            dZ: -- gradient of the cost with respect to Z
        """
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

    def __ReLU_gradient(self, dA, Z):
        """
        Calculates ReLU function gradient

        :param dA: -- post-activation gradient
        :param A: -- linear activation function cache 'Z'
        :returns dZ: -- gradient of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def __tanh_gradient(self, dA, Z):
        """
        Calculates hyperbolic tangent function gradient

        :param dA: -- post-activation gradient
        :param A: -- post-activation gradient
        :returns dZ: -- gradient of the cost with respect to Z
        """
        dZ = dA * (1 - np.power(np.tanh(Z), 2))
        return dZ

    def __LReLU_gradient(self, dA, Z):
        """
        Calculates leaky rectified linear unit function gradient

        :param dA: -- post-activation gradient
        :param Z: -- post-activation gradient
        :returns dZ: -- gradient of the cost with respect to Z
        """
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] *= 0.001
        return dZ

    def __gradient_descent(self, dA, layer_index, activation_type):
        """
        Gradient descent step for a backward propagation step

        :param dA: -- post-activation gradient for current layer l
        :param linear_cache:
        :param activation_cache:
        :param activation_type:
        :returns
            dA_prev: -- Gradient of the cost with respect to the activation
            dW: -- Gradient of the cost with respect to W (current layer l), same shape as W
            db: -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear = self._linear_cache[layer_index - 1]
        if activation_type == Actvitaion_Function.SIGMOID:
            dZ = self.__sigmoid_gradient(dA, linear)
        elif activation_type == Actvitaion_Function.ReLU:
            dZ = self.__ReLU_gradient(dA, linear)
        elif activation_type == Actvitaion_Function.TANH:
            dZ = self.__tanh_gradient(dA, linear)
        elif activation_type == Actvitaion_Function.LReLU:
            dZ = self.__LReLU_gradient(dA, linear)
        else:
            raise ValueError('Provide a valid activation function type: either ReLU, LReLU, sigmoid or tanh')
        dA_prev, dW, db = self.__derivation(dZ, layer_index)
        return dA_prev, dW, db

    def backward_propagation(self, Y):
        """
        Backward propagation step for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        :param layer_index:
        :param Y: -- true "label" vector (i.e. true/false)
        :returns grads: -- a dictionary with gradients
        """
        grads = {}
        iters, l = self._depth - 1, self._depth
        # get current layer cached activation values
        A = self._activation_cache[iters]
        Y = Y.reshape(A.shape)

        # calculate output layer's(sigmoid) activation f-n derivative
        dA = -Y / A + (1 - Y) / (1 - A)

        act_type = Actvitaion_Function.SIGMOID
        grads['dA' + str(l)], grads['dW' + str(l)], grads['db' + str(l)] = self.__gradient_descent(dA, l, act_type)
        dA_prev = grads['dA' + str(l)]

        for l in reversed(range(iters)):
            layer = l + 1
            dA_prev_tmp, dW_tmp, db_tmp = self.__gradient_descent(dA_prev,
                                                                  layer,
                                                                  activation_type=Actvitaion_Function.ReLU)
            grads['dA' + str(layer)], grads['dW' + str(layer)], grads['db' + str(layer)] = dA_prev_tmp, dW_tmp, db_tmp
            dA_prev = dA_prev_tmp

        self._activation_cache.clear()
        self._linear_cache.clear()
        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent

        :param grads: -- gradients
        :param learning_rate: -- size of a gradient descent step
        """
        # Update rule for each parameter. Use a for loop.
        for l in range(1, self._depth + 1):
            W, b, dW, db = self._parameters['W' + str(l)], self._parameters['b' + str(l)], grads['dW' + str(l)], grads[
                'db' + str(l)]
            self._parameters['W' + str(l)] = W - learning_rate * dW
            self._parameters['b' + str(l)] = b - learning_rate * db

    def predict(self, test_set, labels, print_results=True):
        """
        This function is used to predict the results of a multilayer neural network.

        :param test_set: -- a data set that is used to measure the DNN accuracy
        :param labels: -- correct answers for the given test dataset
        :returns
            p: -- predictions for the given dataset
            accuracy:  -- the accuracy of this particular trained model
        """
        m = test_set.shape[1]
        predictions = np.zeros((1, m))

        # Forward propagation
        dnn_outputs = self.forward_propagation(test_set)

        for i in range(0, dnn_outputs.shape[1]):
            if dnn_outputs[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0
        accuracy = np.sum((predictions == labels) / m)

        if print_results:
            print('Accuracy: ', accuracy)
        return predictions, accuracy


def main():
    dnn_model = Deep_Neural_Network([3, 2, 1])
    depth = dnn_model.depth
    depth = 8

if __name__ == '__main__':
    main()