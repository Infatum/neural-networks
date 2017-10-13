import time
import data_resolver
import deep_neural_network as DNN
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np


class Model_Type(Enum):
    Logistic_Regression = 0,
    DNN = 1,
    CNN = 2,

class Binary_Classifier:

    def __init__(self, layers_dimensions, model_type=Model_Type.Logistic_Regression, learning_rate=0.0075, number_of_iterations=3000, print_cost=True):

        self._costs = []
        self._model_type = model_type
        self._learning_rate = learning_rate
        self._numb_of_iter = number_of_iterations
        self._print_cost = print_cost
        self._model = self.init_model(layers_dimensions, Model_Type.DNN)

    @property
    def model(self):
        return self._model

    def init_model(self, layers_dims=0, model_type=Model_Type.Logistic_Regression):
        """
        Initialize predictive model for a binary classification problem

        :param layers_dims: -- layers structure(list that contains values with numbers of neurons on each layer)
        :param model_type: -- type of a predictive model
        :return:
        """
        model = None
        if model_type == Model_Type.DNN:
            if layers_dims != 0 and len(layers_dims) > 0:
                model = DNN.Deep_Neural_Network(layers_dims)
            else:
                raise ValueError('Please, provide a list with layers dimensions')
        else:
            raise NotImplemented('Haven''t implemented yet, sorry folks')
        return model

    def train_model(self, data_manager):
        for i in range(0, self._numb_of_iter):
            model_output = self._model.forward_propagation(data_manager.train_image_data)
            cost = self._model.compute_cost(model_output, data_manager.train_label_data)
            grads = self._model.backward_propagation(data_manager.train_label_data)
            self._model.update_parameters(grads, self._learning_rate)

            # Print the cost every 100 training example
            if self._print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if self._print_cost and i % 100 == 0:
                self._costs.append(cost)

        plot = plt.plot(np.squeeze(self._costs))
        plot.ylabel('cost')
        plot.xlabel('iterations (per tens)')
        plt.title('Learning rate =' + str(self._learning_rate))
        plt.show()

    def predict(self, data_resolver, dataset_type='train', print_results=True):
        if dataset_type == 'train':
            predictions, accuracy = self._model.predict(data_resolver.train_image_data, data_resolver.train_label_data)
        elif dataset_type == 'test':
            predictions, accuracy = self._model.predict(data_resolver.test_image_data, data_resolver.test_label_data)
        else:
            raise NotImplemented('No development set available now. Please enter correct values: train or test')


def main():
    bin_classifier = Binary_Classifier((12288, 20, 7, 5, 1), Model_Type.DNN, 0.01, 2500, print_cost=True)
    data_manager = data_resolver.Data_Resolver(True)
    bin_classifier.train_model(data_manager)
    bin_classifier.predict(data_manager, 'train', True)
    bin_classifier.predict(data_manager, 'test', True)


if __name__ == '__main__':
    main()

