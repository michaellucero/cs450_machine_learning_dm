import numpy as np
import random
from collections import Counter
import math


class Perceptron:
    """
    Perceptron
    A single node with n weights according the number of inputs
    plus the bias node which is weight [0]
    """
    def __init__(self):
        
        self.weights = []
        
        # account for bias
        self.weights.append(random.random())

    def set_weights(self, input_size):
        
        # crate a weight for every input
        for index in range(input_size):
            self.weights.append(random.uniform(-2,2))

    def calculate_weights_activation(self, inputs_Xi):
        
        # initalize output by fist calculating with bias
        h = self.weights[0] * -1

        # Take every weight except bias and multiply and sum to h
        for weight_i, inputs_i in zip(self.weights[2:], inputs_Xi):
            h += (weight_i * inputs_i)
        
        print("Output calculated before threshold:", h)
        return h

    def update_weights(self, inputs_Xi):
        pass


class Layer:
    """
    Layer
    Represents a single layer in a neural network where it 
    can include n perceptron(nodes)
    """
    def __init__(self, n_perceptrons_count):
        self.layer = []
        self.outputs = []

        ###### Threshold might be a per Network value ########
        self.threshold = 0

        # create layer of n length
        for node in range(0, n_perceptrons_count):
            self.layer.append(Perceptron())

    def set_all_perceptron_weights(self, input_size):
        # go through all nodes
        for node in self.layer:
            node.set_weights(input_size)

    def activate_perceptrons(self, inputs):
        
        # go through all nodes
        for node in self.layer:
            #  use activation function h = Sumation wi xi
            h = node.calculate_weights_activation(inputs)
            if(h <= self.threshold):
                # don't fire
                ###### add later adjust weights #####
                self.outputs.append(0)
            else:
                # fired!
                self.outputs.append(1)

        print("Outputs of each perceptron after threshold", self.outputs)
        print()

        ##### For now just clear output but when multilayer change ###
        self.outputs.clear()


class NeuralNetModel:
    """
    Neural Net Model
    """
    def __init__(self, neural_net):
        self.neural_net = neural_net

    def predict(self, testing_data):

        predicted_targets = []
        
        
        ####### TO DO - add predicted from model #########

        
        return predicted_targets


class NeuralNetClassifier:
    """
    Neural Net Classifier
    """	
    def __init__(self, n_perceptrons):
        # Create layer with specified n nodes
        self.single_layer = Layer(n_perceptrons)


    def fit(self, training_data, training_targets):
       
        # get the number of attributes that will make up inputs
        input_size = len(training_data[0])
        # initialize weights for inputs
        self.single_layer.set_all_perceptron_weights(input_size)

        for index, node in enumerate(self.single_layer.layer):
            print("Perceptron", index + 1)
            print("Weights 0 ...", input_size)
            print(node.weights)
            print()

        print(self.single_layer.layer)
        print()

        # process training data one data item at a time
        for data in training_data:
            self.single_layer.activate_perceptrons(data)

# self.hard_coded_model = NeuralNetModel(neural_net)

# return self.hard_coded_model