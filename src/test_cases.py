import numpy as np


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
