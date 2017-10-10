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
        self._activation_cache = []
        self._linear_cache = []
        # todo: features
        #self._features = { 'X',  }

        L = len(layers_dims)
        self._depth = L - 1
        np.random.seed(3)
        if layers_dims is not None:
            for l in range(1, L):
                # initialize weights with random values for each hidden layer
                self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
                # initialize biases with zeros for each hidden layer
                self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
                assert (self._parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
                assert (self._parameters['b' + str(l)].shape == (layers_dims[l], 1))

    @property
    def activations(self):
        return self._activation_cache

    # todo: remove after debug
    @activations.setter
    def activations(self, A):
        self._activation_cache = A

    @property
    def depth(self):
        return self._depth

    # todo: remove after debug
    @depth.setter
    def depth(self, d):
        self._depth = d

    @property
    def parameters(self):
        return self._parameters

    # todo: remove after debug
    @parameters.setter
    def parameters(self, p_s):
        self._parameters = p_s

    @property
    def linear_cache(self):
        return self._linear_cache

    # todo: remove after debug
    @linear_cache.setter
    def linear_cache(self, cache):
        self._linear_cache = cache

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
            self._activation_cache.append(A)
            self._linear_cache.append(Z)

        net_output, Z = self.activation(A, self._depth, Actvitaion_Function.SIGMOID)
        self._activation_cache.append(net_output)
        self._linear_cache.append(Z)
        return self._activation_cache

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

    def __derivation(self, dZ, layer_index):
        """
        Linear portion for a backward propagation
        :param dZ: -- Gradient of the cost with respect to the linear output (of current layer l)
        :param activation_cache: -- list of cached activation value for each layer with L-1 indexing
        :returns
            dA_prev: -- Gradient of the cost with respect to the activation (of the previous layer l-1)
            dW: -- Gradient of the cost with respect to W (current layer l)
            db: -- Gradient of the cost with respect to b (current layer l)
        """
        #todo: refactor me
        # linear_cache indexing
        # l = layer_index - 1
        # current layer's activation values
        A = self._activation_cache[layer_index]
        # get previous layer activation values if layer isn't first, either assign prev. activ. to the feature vector
        A_prev = self._activation_cache[layer_index - 1]
        W, b = self._parameters['W' + str(layer_index)], self._parameters['b' + str(layer_index)]
        # amount of neurons in the previous layer
        m = A_prev.shape[1]

        dW = np.dot(1 / m, np.dot(dZ, A_prev.T))
        db = np.dot(1 / m, np.sum(dZ, axis=1, keepdims=True))
        # todo: debug dimensions mismatch
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

        :param layer_index:
        :param Y:
        :param dA_prev:
        :param dW:
        :param db:
        :return:
        """
        grads = {}
        L = self._depth
        # get current layer cached activation values
        A = self._activation_cache[L]
        Y = Y.reshape(A.shape)

        # calculate output layer's(sigmoid) activation f-n derivative
        dA = -Y / A + (1 - Y) / (1 - A)

        act_type = Actvitaion_Function.SIGMOID
        grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = self.__gradient_descent(dA, L, act_type)
        dA_prev = grads['dA' + str(L)]

        for l in reversed(range(L - 1)):
            layer = l + 1
            dA_prev_tmp, dW_tmp, db_tmp = self.__gradient_descent(dA_prev, layer, activation_type=Actvitaion_Function.ReLU)
            grads['dA' + str(layer)], grads['dW' + str(layer)], grads['db' + str(layer)] = dA_prev_tmp, dW_tmp, db_tmp
            dA_prev = dA_prev_tmp

        self._activation_cache.clear()
        self._linear_cache.clear()
        return grads


def L_model_backward_test_case(dnn_model):
    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
   np.array([[ 0.41791293,  1.91720367]]))])
   """
    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z2 = np.random.randn(3,2)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z3 = np.random.randn(1,2)

    linear_cache = [Z2, Z3]
    activation_cache = [A1, A2, AL]

    dnn_model.depth = 2
    dnn_model.linear_cache = linear_cache
    dnn_model.activations = activation_cache
    dnn_model.parameters['W1'], dnn_model.parameters['b1'], dnn_model.parameters['W2'], dnn_model.parameters['b2'] = W1, b1, W2, b2

    return Y


def print_grads(grads):
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dA1 = " + str(
        grads["dA2"]))  # this is done on purpose to be consistent with lecture where we normally start with A0
    # in this implementation we started with A1, hence we bump it up by 1.


def main():
    n_layer_nn = Deep_Neural_Network([4, 3, 1])
    Y = L_model_backward_test_case(n_layer_nn)
    grads = n_layer_nn.backward_propagation(Y)

    print_grads(grads)


if __name__ == '__main__':
    main()