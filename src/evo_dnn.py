import numpy as np
from typing import Callable
import activation_funciton as af


class EvoDNN:
    def __init__(self,
                 feature_number: int,
                 output_size: int,
                 hidden_layers: np.ndarray,
                 output_layer_activation_function: Callable,
                 hidden_layer_functions: list,
                 verbose: bool = False):
        self.feature_number = feature_number
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.output_layer_activation_function = output_layer_activation_function
        self.hidden_layer_functions = hidden_layer_functions \
            if hidden_layer_functions else [af.sigmoid]
        self.fitness = float("-inf")
        self.verbose = verbose

        self.hidden_layer_bias = list()
        self.hidden_layer_functions_idx = list()
        self.input_to_hidden_layer_weight_matrix = None
        self.hidden_layer_to_output_weight_matrix = None
        self.hidden_layer_to_hidden_layer_weight_matrix_list = list()

    def init_parameters(self):
        for node_size in self.hidden_layers:
            self.hidden_layer_bias.append(np.random.uniform(size=node_size))
            self.hidden_layer_functions_idx.append(
                np.random.randint(
                    len(self.hidden_layer_functions),
                    size=node_size))

        self.input_to_hidden_layer_weight_matrix = np.random.uniform(size=(self.feature_number, self.hidden_layers[0]))
        for layer in range(len(self.hidden_layers) - 1):
            self.hidden_layer_to_hidden_layer_weight_matrix_list.append(
                np.random.uniform(size=(
                    self.hidden_layers[layer],
                    self.hidden_layers[layer + 1])))
        self.hidden_layer_to_output_weight_matrix = np.random.uniform(size=(self.hidden_layers[-1], self.output_size))

        if self.verbose:
            print(f"Network bias: \n {self.hidden_layer_bias}"
                  f"Network function index: \n {self.hidden_layer_functions_idx}\n"
                  f"Network input to hidden layer weight matrix shape: "
                  f"{self.input_to_hidden_layer_weight_matrix.shape}\n"
                  f"Network hidden layer to hidden layer length, shape: "
                  f"{len(self.hidden_layer_to_hidden_layer_weight_matrix_list)}, "
                  f"{self.hidden_layer_to_hidden_layer_weight_matrix_list[0].shape}\n"
                  f"Network hidden layer to output weight matrix shape: "
                  f"{self.hidden_layer_to_output_weight_matrix.shape}")


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
