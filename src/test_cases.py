import numpy as np
import base_neural_network as DNN

# todo: Here implement gradient checking
def gradient_cheking(y, theta, forward_prop, back_prop, epsilon=1e-7):
    """

    :param y:
    :param theta:
    :param forward_prop:
    :param back_prop:
    :param epsilon:
    :return:
    """
    J_plus, J_minus = forward_prop(theta + epsilon), forward_prop(theta - epsilon)
    grad_approxim = (J_plus - J_minus) / 2 * epsilon

    gradient = back_prop(y)

    difference = np.linalg.norm(gradient - grad_approxim) / np.linalg.norm(gradient) + np.linalg.norm(grad_approxim)

    if difference < 1e-7:
        return True
    else:
        raise ValueError('The gradient is wrong!')


