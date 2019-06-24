
import random
#import logging
from train import train_and_score
#from keras.layers import LeakyReLU

class Network():
    """Represent a network and let us operate on it.
    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.
        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.loss = 0.0
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.
        Args:
            network (dict): The network parameters
        """
        self.network = network

    def train(self, x_train , y_train ,x_val , y_val, x_test , y_test , batch_size):
        """Train the network and record the accuracy.
        Args:
            dataset (str): Name of dataset to use.
        """
        if self.loss == 0.:
            self.loss = train_and_score(self.network, x_train , y_train ,x_val , y_val, x_test , y_test , batch_size)

    def print_network(self):
        """Print out a network."""
        print(self.network)
        print("Network Loss: %.2f%%" % (self.loss * 100))
