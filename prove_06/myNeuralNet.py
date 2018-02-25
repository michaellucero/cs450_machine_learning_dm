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
        
        self.input_weights = []
        self.h = None
        self.alpha = None
        self.delta = None
        
        # account for bias
        self.input_weights.append(random.random())

    def set_weights(self, input_size):
        
        # crate a weight for every input
        for index in range(input_size):
            self.input_weights.append(random.uniform(-1,1))

    def calculate_weights_activation(self, inputs_Xi):
        
        # initalize output by fist calculating with bias
        h = self.input_weights[0] * -1

        # Take every weight except bias and multiply and sum to h
        for weight_i, inputs_i in zip(self.input_weights[2:], inputs_Xi):
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
    def __init__(self, n_perceptrons_count, is_output_layer=False):
        self.single_layer = []
        self.outputs = []
        self.is_output_layer = is_output_layer

        ###### Threshold might be a per Network value ########
        self.threshold = 0

        # create layer of n length
        for i in range(0, n_perceptrons_count):
            self.single_layer.append(Perceptron())

    # input size is required to make a weight for every input to the left
    #    of the node.
    def set_all_perceptron_weights(self, input_size):
        # go through all nodes
        for node in self.single_layer:
            node.set_weights(input_size)

    def activate_perceptrons(self, inputs):
        
        # go through all nodes
        for node in self.single_layer:
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



class HiddenLayers:
    """
    HiddenLayers
    """
   
    def __init__(self, hidden_layer_size_node_by_layer):
        self.perceptrons_per_layer_count = hidden_layer_size_node_by_layer
        self.hidden_layer_count = len(hidden_layer_size_node_by_layer)
        self.hidden_layer_list = []
        
        # create hidden layers on layer at a time where the perceptron count varies
        #    per layer
        for i in range(0, self.hidden_layer_count):
            self.hidden_layer_list.append(Layer(self.perceptrons_per_layer_count[i]))
    
    def initializes_layer_with_all_perceptron_weights(self, input_size):
        # go through all layers and each layer will set weights randomly
        
        is_first_layer = True

        for i, layer in enumerate(self.hidden_layer_list):
            # only the first layer weights should be the size of the input where
            #    the others should the size of the previous layers node size
            if is_first_layer:
                layer.set_all_perceptron_weights(input_size)
                is_first_layer = False
            else:
                # the -1 is to account for it being the previous layers size
                layer.set_all_perceptron_weights(self.perceptrons_per_layer_count[i-1])


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
    def __init__(self, hidden_layer_arbitary_size_per_layer):
        # Create layer with specified n nodes
        self.hidden_layers = HiddenLayers(hidden_layer_arbitary_size_per_layer) 
        
        # uninitialize until training data is passed to fit function
        self.input_size = None

        self.output_layer = None
        self.output_option_list = None
        self.output_layer_size = None

    def fit(self, training_data, training_targets):
       
        # get the number of attributes that will make up inputs
        self.input_size = len(training_data[0])
        
        # get the number of classification targets and output node layer size
        self.output_option_list = np.unique(training_targets)
        self.output_layer_size = len(self.output_option_list)

        # initialize weights for hidden layer inputs
        #    Because of the variable size of each hidden layer the weigh_inputs should
        #    be based on that fact. The first hidden layer will be based on input_size
        self.hidden_layers.initializes_layer_with_all_perceptron_weights(self.input_size)

        # create output layer
        self.output_layer = Layer(self.output_layer_size, True)

        # SPECIAL CONDITION 
        #    output layers input weights are to be based on the hidden layer 
        #    size before it.
        self.output_layer.set_all_perceptron_weights(self.hidden_layers.perceptrons_per_layer_count[-1])




        # Print out summary of what has been built
        print("Input size: ", self.input_size)
        print()

        print("classification targets (output options): ", self.output_option_list)
        print()
        print("Output size: ", self.output_layer_size)
        print()
        print()

        # PRINT HIDDEN LAYER
        for layer_index, layer in enumerate(self.hidden_layers.hidden_layer_list):
            print("Hidden Layer: ", layer_index + 1)
            print()
            for node_index, node in enumerate(layer.single_layer):
                print("Perceptron:", node_index + 1)
                print("Weights 0(Bias) 1 ...", len(node.input_weights) - 1)
                print(node.input_weights)
                print()

            #print(layer.single_layer)
            print()
        
        print("Output Layer: ")
        print()
        # PRINT OUTPUT LAYER
        for node_index, node in enumerate(self.output_layer.single_layer):
            print("Perceptron:", node_index + 1)
            print("Weights 0(Bias) 1 ...", len(node.input_weights) - 1)
            print(node.input_weights)
            print()

        # process training data one data item at a time
        # for data in training_data:
        #     self.single_layer.activate_perceptrons(data)

# self.hard_coded_model = NeuralNetModel(neural_net)

# return self.hard_coded_model


#  def fit(self, training_data, training_targets):
       
#         # get the number of attributes that will make up inputs
#         input_size = len(training_data[0])
#         # initialize weights for inputs
#         self.single_layer.set_all_perceptron_weights(input_size)

#         for index, node in enumerate(self.single_layer.layer):
#             print("Perceptron", index + 1)
#             print("Weights 0 ...", input_size)
#             print(node.input_weights)
#             print()

#         print(self.single_layer.layer)
#         print()

#         # process training data one data item at a time
#         for data in training_data:
#             self.single_layer.activate_perceptrons(data)

# self.hard_coded_model = NeuralNetModel(neural_net)

# return self.hard_coded_model