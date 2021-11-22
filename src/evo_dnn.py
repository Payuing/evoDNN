import numpy as np
from typing import Callable
import activation_funciton as af


class EvoDNN:
    def __init__(self,
                 feature_number: int,
                 output_size: int,
                 layers: np.ndarray,
                 output_layer_activation_function: Callable,
                 hidden_layer_functions: list,
                 verbose: bool = False):
        self.feature_number = feature_number
        self.output_size = output_size
        self.layers = layers
        self.output_layer_activation_function = output_layer_activation_function
        self.hidden_layer_functions = hidden_layer_functions \
            if hidden_layer_functions else [af.sigmoid]
        self.fitness = float("-inf")
        self.verbose = verbose
        self.first_layer = 0

        self.biases = list()
        self.functions_idxes = list()
        self.weights = list()

    def init_parameters(self):
        for neuron_size in self.layers:
            self.biases.append(np.random.uniform(size=neuron_size))
            self.functions_idxes.append(
                np.random.randint(
                    len(self.hidden_layer_functions),
                    size=neuron_size))

        for layer in range(len(self.layers)):
            if layer == self.first_layer:
                self.weights.append(
                    np.random.uniform(size=(
                        self.feature_number,
                        self.layers[self.first_layer]))
                )
            else:
                self.weights.append(
                    np.random.uniform(size=(
                        self.layers[layer - 1],
                        self.layers[layer])))
        if self.verbose:
            self.show_verbose()

    def show_verbose(self):
        print(f"Network biases: \n {self.biases}"
              f"Network function indexes: \n {self.functions_idxes}\n"
              f"Network weight length, shape: "
              f"{len(self.weights), self.weights[0].shape}\n")


if __name__ == '__main__':
    feature_number_test = 10
    output_size_test = 10
    hidden_layers_test = np.array([10, 10, 10])
    output_layer_activation_function_test = af.softmax
    hidden_layer_functions_test = [af.sigmoid, af.tanh]
    verbose_test = True
    net = EvoDNN(feature_number_test,
                 output_size_test,
                 hidden_layers_test,
                 output_layer_activation_function_test,
                 hidden_layer_functions_test,
                 verbose_test)
    net.init_parameters()
