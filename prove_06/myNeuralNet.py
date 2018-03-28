import numpy as np
import random
from collections import Counter
import math
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt



class Perceptron:
    """
    Perceptron
    A single node with n weights according the number of inputs
    plus the bias node which is weight [0]
    """
    def __init__(self, index):
        
        self.input_weights = []
        self.h = None
        self.alpha = None
        self.delta = None
        # Keep track of index in the node layer, also used for output layer
        self.node_num = index
        
        # account for bias
        self.input_weights.append(random.random())

    def set_weights(self, input_size):
      
        # crate a weight for every input
        for index in range(input_size):
            self.input_weights.append(random.uniform(-1,1))

    def squashing_function(self):
        print("Squashing Function: g(", self.h, ")")
        self.alpha = 1/(1 + math.exp(-1 * (self.h)))
        print("-> alpha: ", self.alpha)
        print()
        

    def calculate_weights_activation(self, inputs_Xi):
        print("inputs: ", inputs_Xi)
        
        print("input weights w/bias: ", self.input_weights)
        
        print("input weights", self.input_weights[1:])


        print("Summation of: " )
        print( "   Bias input: -1 * Weight:", self.input_weights[0])
        # initalize output by fist calculating with bias
        self.h = self.input_weights[0] * -1

        
        # Take every weight except bias and multiply and sum to h
        for weight_i, input_i in zip(self.input_weights[1:], inputs_Xi):
            
            print( "   Input:", input_i, "* Weight:", weight_i)

            self.h += weight_i * float(input_i)

        print("->  h: ",self.h)

        self.squashing_function()


    def calculate_error_output_node(self, output_node):        ######################## output node is the firing index
        
        print("I am output layer node:", self.node_num)
        print("w/ alpha", self.alpha, "and target node number", self.node_num)

        # output layer back propagation calulation dj = aj(1 - aj)(aj - tj)
        ##############self.delta = self.alpha * (1 - self.alpha) * (self.alpha - output_node)###############
        if (output_node == self.node_num):
            self.delta = self.alpha * (1 - self.alpha) * (self.alpha - 0)   ###############
            print("Delta:", self.delta)
            print()
        else:
            self.delta = self.alpha * (1 - self.alpha) * (self.alpha - 1)   ##################
            print("Delta:", self.delta)
            print()


    # NOTE: calculating error for hidden node done in back_propagate becuase requires access to both
    #   the current layer and the layer to the right of it.

    def calculate_new_weight(self, learning_rate, inputs):
        print("inputs:", inputs)
        print("input weights w/ bias:", self.input_weights)
        print("delta:", self.delta)

        for index, weight in enumerate(self.input_weights):
            # account for bias first
            if index == 0:
                print("   weight:", weight, "- Learning rate:", learning_rate, "*")
                print("          current node delta:", self.delta, "* left alpha/input:", -1)
                self.input_weights[index] = weight - (learning_rate * self.delta * (- 1.0))
            else:
                # the inputs start at index 0 but it doesn't include the bias so - 1 fixes
                print("   weight:", weight, "- Learning rate:", learning_rate, "*")
                print("          current node delta:", self.delta, "* left alpha/input:", inputs[index - 1])
                self.input_weights[index] = weight - (learning_rate * self.delta * float(inputs[index - 1]))     

        print("Updated weights", self.input_weights)
        print()


class Layer:
    """
    Layer
    Represents a single layer in a neural network where it 
    can include n perceptron(nodes)
    """
    def __init__(self, n_perceptrons_count=None, output_option_list=None, is_output_layer=False):
        
        self.is_output_layer = is_output_layer

        # keep track if this is an output layer
        # used if output node
        if self.is_output_layer:
            self.output_option_list = None
            self.layer_size = None
            self.firing_node_index = None
        
        self.single_layer = []

        # this is alpha outputs
        self.outputs = []

        # this is delta errors
        self.errors = []
        
        if is_output_layer:
            self.output_option_list = output_option_list
            self.layer_size = len(self.output_option_list)
        else:
            self.layer_size = n_perceptrons_count

        # create layer of n length
        for i in range(0, self.layer_size):
            self.single_layer.append(Perceptron(i))
        
     

    # input size is required to make a weight for every input to the left
    #    of the node.
    def set_all_perceptron_weights(self, input_size):
        # go through all nodes
        for node in self.single_layer:
            node.set_weights(input_size)

    def activate_perceptrons(self, inputs):
        
        if self.is_output_layer:
            print("******* Output Layer ******* ")
            print()
            print("Classes", self.output_option_list)
            print()
        else:
             print("****** Hidden Layer ******* ")
             print()

        # go through all nodes
        for index, node in enumerate(self.single_layer):
            # use activation function h = Sumation wi xi and sigmoid
            # append to keep track of outputs for calculating next layer

            print("Node:", node.node_num)
            node.calculate_weights_activation(inputs)
                
            self.outputs.append(node.alpha)
            

        print("Outputs of each perceptron (alpha)", self.outputs)
        print()
        print()

    def update_weights(self, learning_rate, inputs):
        for index, node in enumerate(self.single_layer):
            
            # calulate weights and update one at a time
            print("Node:", node.node_num)
            node.calculate_new_weight(learning_rate, inputs)
          
            
                
           

    def epoch_reset_single_layer(self):
        self.outputs = []
        self.errors = []
        for node in self.single_layer:
                node.alpha = None
                node.h = None
                node.delta = None
   
    # check if output was accurate by getting the largest output index which matches the index position
    #   of the output options. 
    def is_correct_predictions(self, target):
        print("Possible Classifications:", self.output_option_list)

        # set node that is the output so when calculating output for error on output node can calculate
        print("Highest node value:", max(self.outputs))

        self.firing_node_index = self.outputs.index(max(self.outputs))

        
        print("Predicted firing node:", self.firing_node_index)
        

        predicted_target = self.output_option_list[self.outputs.index(max(self.outputs))]
        
        print("Predicted Target:", predicted_target)
        print("Actual Target:", target)
        if target == predicted_target:
            return True


    def get_firing_node_predicted_target(self):
        predicted_target = self.output_option_list[self.outputs.index(max(self.outputs))]

        print("predicted Target: ",predicted_target)
        print()

        return predicted_target


    def back_propagate(self, layer_to_right=None):

        # special condition with calulating the error of the output layer
        if self.is_output_layer:
            print("**** Calculating output layer errors *****")
            print()
            for node in self.single_layer:
                node.calculate_error_output_node(self.firing_node_index)  ######################################## is this supposed to be self.firing_node_index??? or 
                self.errors.append(node.delta)
        # calulate different for hidden layer
        else:
            for index, left_node in enumerate(self.single_layer):
                sum_error = 0
                print("I'm hidden layer node:", index, "w/ alpha", left_node.alpha)
                print()
                for index2, right_node in enumerate(layer_to_right.single_layer):
                    # get the weight * the delta that corresponds to the output from the left layer from
                    #  the right layer... SUM Wjk djk  -- k meaning layer on right
                    print("I'm right layer node", index2, "w/ weights", right_node.input_weights)
                    print("Using weight:", right_node.input_weights[index + 1], "* delta:", right_node.delta )
                    print()
                    sum_error += right_node.delta * right_node.input_weights[index + 1]
                
                # Perform calculation aj(1- aj)SUM Wjk djk   -- k meaning layer on right
                
                left_node.delta = left_node.alpha * (1 - left_node.alpha) * sum_error   
                print("hidden layer calculated error (delta) -> ", left_node.delta)
                print() 
                self.errors.append(left_node.delta)
            

        print("Errors of each perceptron in layer (delta):", self.errors)
        print()
        print()

 



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

    def activate_layers(self, inputs):

        previous_layer_output = []
        # compute h for each node
        for index, layer in enumerate(self.hidden_layer_list):
            # input layer is first
            if index == 0:
                layer.activate_perceptrons(inputs)
                previous_layer_output = layer.outputs
            else: # followed by all hidden layers
                layer.activate_perceptrons(previous_layer_output) 
                # after using last layer clear for new data
                previous_layer_output = []
                previous_layer_output = layer.outputs
        
    
    def back_propagate_layers(self, output_layer, inputs):    ################ what is inputs used for 
        layer_to_right = None

        print("**** Calculating hidden layer errors *****")
        print()
        for index, layer in reversed(list(enumerate(self.hidden_layer_list))):
            # starting from output layer first
            print("I'm hidden layer:", index + 1, "of", self.hidden_layer_count, "hidden layers")
        
            if index + 1 == self.hidden_layer_count:
                # give the hidden layer the output layer
                layer.back_propagate(output_layer)
                # after using last layer clear for new data
                layer_to_right = None
                # keep track of the last layer
                layer_to_right = layer

            # followed by all hidden layers
            else:
                layer.back_propagate(layer_to_right) 
                # after using last layer clear for new data
                layer_to_right = None
                # keep track of the last layer
                layer_to_right = layer

    def update_layer_weights(self, learning_rate, inputs):

        print("******** Updating Weights *********")
        previous_layer_output = []
        # update weights one layer at a time
        for index, layer in enumerate(self.hidden_layer_list):
            print("** Hidden Layer:", index, "**")
            print()
            # input layer is first
            if index == 0:
                layer.update_weights(learning_rate, inputs)
                previous_layer_output = layer.outputs
            else: # followed by all hidden layers
                layer.update_weights(learning_rate, previous_layer_output) 
                # after using last layer clear for new data
                previous_layer_output = []
                previous_layer_output = layer.outputs


    def epoch_reset_hidden_layers(self):
        for layer in self.hidden_layer_list:
            layer.epoch_reset_single_layer()
            
            
       


class NeuralNetModel:
    """
    Neural Net Model
    """
    def __init__(self, hidden_layers, output_layer):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def predict(self, testing_data):

        predicted_targets = []
        
        # loop through each data point and do feed forward
        for inputs in testing_data:

            print("################ New Data Point ################")
            print()

            self.hidden_layers.activate_layers(inputs)

            # activate the output layer based on the last hidden layers outputs
            self.output_layer.activate_perceptrons(self.hidden_layers.hidden_layer_list[self.hidden_layers.hidden_layer_count - 1].outputs)
            
            # get predicted target
            predicted_targets.append(self.output_layer.get_firing_node_predicted_target())
            
            # reset feed forward calulations for next input set
            self.hidden_layers.epoch_reset_hidden_layers()
            self.output_layer.epoch_reset_single_layer()

        
        ####### TO DO - add predicted from model #########

        
        return predicted_targets



class NeuralNetClassifier:
    """
    Neural Net Classifier
    """	
    def __init__(self, hidden_layer_arbitary_size_per_layer, learning_rate):
       
        # Create layer with specified n nodes
        self.hidden_layers = HiddenLayers(hidden_layer_arbitary_size_per_layer) 
        
        # uninitialize until training data is passed to fit function
        self.input_size = None

        self.output_layer = None
        self.learning_rate = learning_rate

        self.epoch_count = 0

    def train_neural_net(self, training_data, training_targets):
         # get the number of attributes that will make up inputs
        self.input_size = len(training_data[0])
        
        # get the number of classification targets and output node layer size
        #self.output_option_list = np.unique(training_targets)
        #self.output_layer_size = len(self.output_option_list)

        # initialize weights for hidden layer inputs
        #    Because of the variable size of each hidden layer the weigh_inputs should
        #    be based on that fact. The first hidden layer will be based on input_size
        self.hidden_layers.initializes_layer_with_all_perceptron_weights(self.input_size)

        # create output layer
        self.output_layer = Layer(None, np.unique(training_targets), True)

        # SPECIAL CONDITION 
        #    output layers input weights are to be based on the last hidden layer 
        #    size before it.
        self.output_layer.set_all_perceptron_weights(self.hidden_layers.perceptrons_per_layer_count[-1])

        data_set_length = len(training_targets)
        correct_count = 0
        wrong_count = 0
        x = []
        y = []

        for i in range(1, 5):
            # loop through each data point and do feed forward
            for inputs, target in zip(training_data, training_targets):

                print("################ New Data Point ################")
                print()

                self.hidden_layers.activate_layers(inputs)

                # activate the output layer based on the last hidden layers outputs
                self.output_layer.activate_perceptrons(self.hidden_layers.hidden_layer_list[self.hidden_layers.hidden_layer_count - 1].outputs)
                
                # if not correct then update weights by calculating dalta
                if(self.output_layer.is_correct_predictions(target)):
                    print("Predicted Right!!!!!! - Go to next epoch:")
                    print()
                    print()
                    correct_count += 1
                else:
                    print("Predicted Wrong!!!!!! - Calculating Error for back-propagation:")
                    print()
                    print()
                    wrong_count += 1
                    
                    # start back propagation by calulating deltas
                    self.output_layer.back_propagate()
                    self.hidden_layers.back_propagate_layers(self.output_layer, inputs) ############# does inputs get used here

                    # finaly update all weights    
                    self.hidden_layers.update_layer_weights(self.learning_rate, inputs)
                    print("** Output Layer **")
                    print()
                    self.output_layer.update_weights(self.learning_rate, self.hidden_layers.hidden_layer_list[self.hidden_layers.hidden_layer_count - 1].outputs)
                    
                
                    
                # reset the alpha, outputs, delta, h everything for the next epoch
                self.hidden_layers.epoch_reset_hidden_layers()
                self.output_layer.epoch_reset_single_layer()
                self.epoch_count += 1
                
            x.append(self.epoch_count)
            y.append(correct_count / (correct_count + wrong_count))
            correct_count = 0
            wrong_count = 0
            
            print(training_data, training_targets)
            
            training_data, training_targets = shuffle(training_data, training_targets)
            
            print(training_data, training_targets)

            print()
            print()
            print("Epochs trained on:", self.epoch_count)
            plt.plot(x,y)
            #plt.ylim([0,1])
            plt.xlabel('epoch count')
            plt.ylabel('correct ratio')
        plt.show()
        print(x)
        print(y)
    
    def fit(self, training_data, training_targets):
        
       
        self.train_neural_net(training_data, training_targets)

        model = NeuralNetModel(self.hidden_layers, self.output_layer) 

        return model
