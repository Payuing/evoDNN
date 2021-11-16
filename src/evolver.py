import time
import random

import numpy as np
import activation_funciton as ac
import loss_function
import torch.nn as nn
import torch
from typing import Callable


class Evolver:
    def __init__(self,
                 max_iter: int = 10,
                 early_stopping: int = 10,
                 layers: np.ndarray = np.array([10], dtype = np.uint32),
                 parent_size: int = 10,
                 children_size: int = 10,
                 weight_mutation_probability: float = 0.1,
                 function_mutation_probability: float = 0.1,
                 weight_mutation_radius: float = 1.0,
                 crossover_proportion: float = 0.5,
                 bias_mutation_probability: float = 0.01,
                 bias_mutation_radius: float = 1.0,
                 is_elitism: bool = True,
                 tournament_size: int = 2,
                 fitness_function: Callable = loss_function.rmse,
                 output_layer_activation_function: Callable = ac.softmax,
                 candidate_function_dict: dict = {0: ac.sigmoid},
                 is_random_state: bool = False,
                 verbose: bool = False):
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.layers = layers
        self.parent_size = parent_size
        self.children_size = children_size
        self.weight_mutation_probability = weight_mutation_probability
        self.function_mutation_probability = function_mutation_probability
        self.weight_mutation_radius = weight_mutation_radius
        self.crossover_proportion = crossover_proportion
        self.bias_mutation_probability = bias_mutation_probability
        self.bias_mutation_radius = bias_mutation_radius
        self.is_elitism = is_elitism
        self.tournament_size = tournament_size
        self.fitness_function = fitness_function
        self.output_layer_activation_function = output_layer_activation_function
        self.candidate_function_dict = candidate_function_dict
        self.is_random_state = is_random_state
        self.verbose = verbose

        if self.is_random_state:
            seed = int(time.time())
            np.random.seed(seed)
            random.seed(seed)
            torch.random.manual_seed(seed)

        print(f"Network has {len(self.layers)} layers. The topology is {self.layers}")

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
              x_validate: np.ndarray = None, y_validate: np.ndarray = None):
        if self.verbose:
            print(f"Input shape is {x_train.shape[0]} X {x_train.shape[1]}.")
            if x_validate:
                print(f"Validation shape is {x_validate.shape[0]} X {x_validate.shape[0]}")

