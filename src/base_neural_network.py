import numpy as np
from dnn_types import Actvitaion_Function
from dnn_types import Initialization_Type
from dnn_types import NN_Mode

class Base_Neural_Network:

    def __init__(self, layers_dims, layers_activations, mode, init_type=Initialization_Type.random, factor=0.001):
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
        self._features = None
        self._layers_activations = {}

        if mode == NN_Mode.Binary_Classification:
            self._nn_mode = NN_Mode.Binary_Classification
        elif mode == NN_Mode.Multiclass_Classification:
            self._nn_mode = NN_Mode.Multiclass_Classification
        else:
            self._nn_mode = NN_Mode.Regression

        L = len(layers_dims)
        self._depth = L - 1
        if init_type == Initialization_Type.random:
            self._initialize_network(layers_dims, layers_activations, factor)


    def _initialize_network(self, layers_dims, layers_activation_functions, factor):
        """
        Random init of a Neural Networks weights and init with zeros for biases

        :param layers_dims: -- layers dimensions description (amount of hidden units)
        :param layers_activation_functions: -- layers activation function descrpition
        :param factor:
        :return:
        """
        print('Base Neural Network init')
        for l in range(1, self._depth + 1):
            self._parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * factor
            self._parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
            self._layers_activations[str(l)] = layers_activation_functions[layers_activation_functions - 1]

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

        :param previous_activation: -- activations from previous layer
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
        elif activation_type == Actvitaion_Function.LINEAR_REGRESSION:
            raise ValueError('Linear regression doesn''t have a non-linear activation part')
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
        A = X
        self._features = X

        for l in range(1, self._depth):
            A_prev = A
            if self._layers_activations != Actvitaion_Function.LINEAR_REGRESSION:
                A, Z = self.activation(A_prev, l, self._layers_activations[l])
                self._activation_cache.append(A)
            else:
                Z = self.__linear_forward(A, l)
            self._linear_cache.append(Z)

        if self._layers_activations[self._depth] == Actvitaion_Function.LINEAR_REGRESSION:
            net_output = Z
        else:
            net_output, Z = self.activation(A, self._depth, self._layers_activations[self._depth])
            self._activation_cache.append(net_output)
        self._linear_cache.append(Z)

        assert (net_output.shape == (1, X.shape[1]))
        return net_output

    def compute_loss(self, net_output, Y):
        out_activ = self._layers_activations[self._depth]
        loss = None
        if out_activ == Actvitaion_Function.SIGMOID and self._nn_mode is NN_Mode.Binary_Classification:
            # cross-entropy loss for the binary classification problem
            loss = -np.dot(Y, np.log(net_output).T) - np.dot(1 - Y, np.log(1 - net_output).T)
        elif out_activ == Actvitaion_Function.SOFTMAX and self._nn_mode is NN_Mode.Multiclass_Classification:
            # cross-entropy loss for the multiclass classification problem
            loss = np.sum(np.dot(Y, np.log(net_output).T))
        elif out_activ == Actvitaion_Function.LINEAR_REGRESSION and self._nn_mode.Regression:
            loss = np.sum(np.square(Y - net_output))
        else:
            raise NotImplementedError('Not implemented yet')
        return loss

    def compute_cost(self, net_output, Y):
        """
        Computes the cost over all training set

        :param net_output:
        :param Y: -- true "label" vector
        :returns cost: -- cross-entropy cost(a mean value for the cross-entropy loss over all training examples)
        """
        # m = amount of training examples
        m = Y.shape[1]
        if self._nn_mode is NN_Mode.Regression:
            cost = (1. / (2*m)) * self.compute_loss(net_output, Y)
        else:
            cost = (1. / m) * self.compute_loss(net_output, Y)
        cost = np.squeeze(cost)

        return cost

    def __derivation(self, dZ, layer_index):
        """
        Computes weights, biases and previous activations gradients

        :param dZ: -- Gradient of the cost with respect to the linear output (of current layer l)
        :param activation_cache: -- list of cached activation value for each layer with L-1 indexing
        :returns
            dA_prev: -- Gradient of the cost with respect to the activation (of the previous layer l-1)
            dW: -- Gradient of the cost with respect to W (current layer l)
            db: -- Gradient of the cost with respect to b (current layer l)
        """
        A = self._activation_cache[layer_index - 1]
        # get previous layer activation values if layer isn't first, either assign prev. activ. to the feature vector
        A_prev = self._activation_cache[layer_index - 2] if layer_index > 1 else self._features
        W, b = self._parameters['W' + str(layer_index)], self._parameters['b' + str(layer_index)]
        # amount of neurons in the previous layer
        m = A_prev.shape[1]

        dW = np.dot(1. / m, np.dot(dZ, A_prev.T))
        db = np.dot(1. / m, np.sum(dZ, axis=1, keepdims=True))
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
        s = self.sigmoid(Z)
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

    def _loss_gradient(self, A, Y):
        out_activ = self._layers_activations[self._depth]
        loss_grad = None
        if out_activ == Actvitaion_Function.SIGMOID and self._nn_mode is NN_Mode.Binary_Classification:
            # cross-entropy loss gradient
            loss_grad = -Y / A + (1 - Y) / (1 - A)
        elif out_activ == Actvitaion_Function.SOFTMAX and self._nn_mode is NN_Mode.Multiclass_Classification:
            # softmax cross-entropy loss gradient
            loss_grad = A - Y
        else:
            loss_grad = np.dot(np.sum(A - Y), self._features)
        return loss_grad

    def _compute_gradients(self, dA, layer_index, activation_type):
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
        elif activation_type == Actvitaion_Function.LINEAR_REGRESSION:
            dZ = 1
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

        # cross-entropy loss derivative
        dA = self._loss_gradient(A, Y)

        act_type = self._layers_activations[self._depth]
        grads['dA' + str(l)], grads['dW' + str(l)], grads['db' + str(l)] = self._compute_gradients(dA, l, act_type)

        # check whether using base model or a derivative one
        if self.__class__.__name__ == 'Base_Neural_Network':
            dA_prev = grads['dA' + str(l)]
            for l in reversed(range(iters)):
                layer = l + 1
                dA, dW, db = self._compute_gradients(dA_prev, layer, self._layers_activations[l])
                grads['dA' + str(layer)], grads['dW' + str(layer)], grads['db' + str(layer)] = dA, dW, db
                dA_prev = dA

            self._activation_cache.clear()
            self._linear_cache.clear()
        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent

        :param grads: -- gradients
        :param learning_rate: -- size of a gradient descent step
        """
        if self._nn_mode is NN_Mode.Binary_Classification or self._nn_mode is NN_Mode.Multiclass_Classification:
            for l in range(1, self._depth + 1):
                W, b, dW, db = self._parameters['W' + str(l)], self._parameters['b' + str(l)], grads['dW' + str(l)], grads[
                    'db' + str(l)]
                self._parameters['W' + str(l)] = W - learning_rate * dW
                self._parameters['b' + str(l)] = b - learning_rate * db


    def predict(self, test_set, labels):
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

        dnn_outputs = self.forward_propagation(test_set)

        for i in range(0, dnn_outputs.shape[1]):
            if dnn_outputs[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0
        accuracy = np.sum((predictions == labels) / m)
        return predictions, accuracy