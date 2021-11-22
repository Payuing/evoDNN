import activation_funciton as af
import evo_dnn
import numpy as np
import time

seed = int(time.time())
np.random.seed(seed)


class Crossover:
    def __init__(self, x_net: evo_dnn.EvoDNN, y_net: evo_dnn.EvoDNN):
        self.x_net = x_net
        self.y_net = y_net
        self.offspring = evo_dnn.EvoDNN(
            self.x_net.feature_number,
            self.x_net.output_size,
            self.x_net.layers,
            self.x_net.output_layer_activation_function,
            self.x_net.hidden_layer_functions)

        for layer in range(len(self.offspring.layers)):
            probability_matrix = np.random.uniform(size=self.offspring.layers[layer])

            self._crossover_activation_functions(layer, probability_matrix)

            self._crossover_biases(layer, probability_matrix)

            self._crossover_weights(layer, probability_matrix)

    def _crossover_activation_functions(self, layer: int, probability_matrix: np.ndarray):
        self.offspring.functions_idxes.append(
            np.zeros(self.offspring.layers[layer]))
        self.offspring.functions_idxes[-1][probability_matrix < 0.5] = \
            self.x_net.functions_idxes[layer][probability_matrix < 0.5]
        self.offspring.functions_idxes[-1][probability_matrix >= 0.5] = \
            self.y_net.functions_idxes[layer][probability_matrix >= 0.5]

    def _crossover_biases(self, layer: int, probability_matrix: np.ndarray):
        self.offspring.biases.append(
            np.zeros(self.offspring.layers[layer]))
        self.offspring.biases[-1][probability_matrix < 0.5] = \
            self.x_net.biases[layer][probability_matrix < 0.5]
        self.offspring.biases[-1][probability_matrix >= 0.5] = \
            self.y_net.biases[layer][probability_matrix >= 0.5]

    def _crossover_weights(self, layer: int, probability_matrix: np.ndarray):
        if layer == self.offspring.first_layer:
            feature_number = self.offspring.feature_number
        else:
            feature_number = self.offspring.layers[layer - 1]
        self.offspring.weights.append(np.zeros((
            feature_number, self.offspring.layers[layer])))
        for feature in range(self.offspring.feature_number):
            self.offspring.weights[-1][feature][probability_matrix < 0.5] = \
                self.x_net.weights[layer][feature][probability_matrix < 0.5]
            self.offspring.weights[-1][feature][probability_matrix >= 0.5] = \
                self.y_net.weights[layer][feature][probability_matrix >= 0.5]

    def get_offspring(self):
        return self.offspring







if __name__ == '__main__':
    feature_number_test = 10
    output_size_test = 10
    hidden_layers_test = np.array([10, 10, 10])
    output_layer_activation_function_test = af.softmax
    hidden_layer_functions_test = [af.sigmoid, af.tanh]
    verbose_test = True
    x_net_test = evo_dnn.EvoDNN(feature_number_test,
                           output_size_test,
                           hidden_layers_test,
                           output_layer_activation_function_test,
                           hidden_layer_functions_test,
                           verbose_test)
    x_net_test.init_parameters()
    y_net_test = evo_dnn.EvoDNN(feature_number_test,
                           output_size_test,
                           hidden_layers_test,
                           output_layer_activation_function_test,
                           hidden_layer_functions_test,
                           verbose_test)
    y_net_test.init_parameters()
