import numpy as np
from collections import Counter
import math

class Node:
	def __init__(self):
		self.attribute_name = -1
		self.possible_children = []
		self.child_list = []
		self.answer = -1

	def update_possible_children(self, smallest_entropy_index):
		self.possible_children = np.delete(self.possible_children, smallest_entropy_index)
		self.attribute_name = smallest_entropy_index		
		print("Current node (Parent): ", self.attribute_name)
		print("Possible Children:", self.possible_children)
		print()
	
	# def add_children(self, index, child):
	# 	self.children[index] = child

class DecisionTreeModel:
	"""
	Decision Tree Model
	"""
	def __init__(self, decision_tree):
		self.decision_tree = decision_tree
	
	def predict(self, testing_data):

		predicted_targets = []
		
		for test in testing_data:
			predicted_targets.append(0)
		return predicted_targets

class DecisionTreeClassifier:
	"""
	Decision Tree classifier
	"""	
	def __init__(self):
		pass
		#self.attributes = {}
		#self.is_root = true


	def calcluate_weighted_entropy(self, attribute_column, targets):
		
		# get choices availiable to be assigned in each column 
		# (np.unique POSSIBLE PROBLEM at it CHNAGES ORDER by sorting but may not cuase bug)
		attribute_options = np.unique(attribute_column)

		options_entropy_list = []

		# compute one columns entropy each loop
		for option in attribute_options:
			feature_list = []
			
			# column values check, record "index - target"
			for index, row_value in enumerate(attribute_column):
				
				# where matched is counted in entropy calculation for single attribute column
				if (row_value == option):
					feature_list.append([index + 1, targets[index]])

			feature_list = np.asarray(feature_list)
				
			print("Feature: ", option, " Targets:", feature_list[:,1])
			# calculate entropy -(option A/count in col * log_2(option A/count in col)) -
			# -(option B/count in col * log_2(option B/count in col))
			# count the instances of each value happening
			iProbability = Counter(feature_list[:,1])
			
			print("target: count of target")
			print(iProbability)
			print()

			# calulate the entropy of each option availiable in the column
			target_option_entropy_list = []
			for key in iProbability:

				# calulate part of the formula so summing can be done for the nth amount of target options
				target_option_entropy_list.append(-(iProbability[key]/len(feature_list)) * math.log2(iProbability[key]/len(feature_list)))
			
			# sum up different options calculations to get entropy 
			options_entropy_list.append(sum(map(float, target_option_entropy_list)))
		
		# Sum up all columns (attributes) and take weighted average to calulate entropy mean of column
		return sum(map(float, options_entropy_list)) / len(attribute_options) 
	



	
	def build_tree(self, training_data, training_targets, available_children):
		
		current_node = Node()

		# set possible node values/attributes
		# if root node then start with all options possible
		#if(self.is_root == true):
		#	current_node.possible_children = available_children
		#	self.is_root  = false
		#else:
		current_node.possible_children = available_children
		


		weighted_average_entropy_list = []
		# select a column to calulate entropy
		for index, attribute_column in enumerate(training_data.T):
			
			#check if leaf nodes we are working with by checking if column has same values
			if(1 == len(np.unique(attribute_column))):
				leaf = Node()
				leaf.answer = attribute_column[0]
				current_node.child_list.insert(index, leaf)
			
			elif (index in current_node.possible_children):
				print("Attribute Column: ", index)
				print(attribute_column)
				print()

				# take one column at a time and calculate entropy and store results
				weighted_average_entropy_list.append(self.calcluate_weighted_entropy(attribute_column, training_targets))

				print("Calculated Entropy list:")
				print(weighted_average_entropy_list)
				print()
				
		if(len(current_node.possible_children) == 0):
			return current_node

		# choose the lowest entropy to spit on - index lines up with order 
		smallest_entropy = list([weighted_average_entropy_list.index(min(weighted_average_entropy_list)), min(weighted_average_entropy_list)])

		# split on smallest entropy 1st elt is index of column in training data, 2nd elt value
		print("smallest entropy (attribute column index, value):")
		print(smallest_entropy)
		
		#print(training_data)
		print()
		#print(training_data.T[smallest_entropy[0]])
		
		#set root node to best attribute
		current_node.update_possible_children(smallest_entropy[0])
		#current_node. = Node(self.update_possible_attribute(smallest_entropy[0]))
		

		# get attribute options
		attribute_options = np.unique(training_data[current_node.attribute_name])

		#subset_filter = training_data[:, current_node == attribute_options[0]]

		#print(subset_filter)
		
		# split and create children
		for option in attribute_options:
			feature_list = []
			sub_data_list = []
			sub_target_list = []
			


			# column values check, record "index - target" that make sublist
			for index, row_value in enumerate(training_data.T[current_node.attribute_name]):
				
				# where matched create sublist
				if (row_value == option):
					sub_data_list.append(training_data[index])
					sub_target_list.append(training_targets[index])
					feature_list.append([index + 1, training_targets[index]])

			feature_list = np.asarray(feature_list)
			sub_data_list = np.asarray(sub_data_list)
			sub_target_list = np.asarray(sub_target_list)
			print("row - target")
			print(feature_list)
			print()
			print("Sublist based on rows - targets")
			print("Parent", current_node.attribute_name, "option", option, "of", attribute_options)
			print("data:", sub_data_list)
			print("targets:", sub_target_list)

		
			# Recurse to sublist but check if leaf before
			#if(0 == len(current_node.possible_children)): #len(self.attributes) > 0):
				# put in the result that is the target
			#	current_node.answer = sub_target_list[0]
			#else:	
			print()

			current_node.child_list.insert(option, self.build_tree(sub_data_list, sub_target_list, current_node.possible_children))
			#current_node.add_children(option, child)
			#current_node .children["option"] = [child]
			#current_node.children.append(child)
			#current_node.children.insert(option, child)
			#current_node.children.append
			#current_node.children().append(child)
			print(current_node)
			print(current_node.child_list)
			
			
		return current_node
		
		#decision_tree[smallest_entropy[0]] = training_data


	def fit(self, training_data, training_targets):
		

		
		decision_tree = self.build_tree(training_data, training_targets, np.arange(training_data.shape[1]))

		print(decision_tree)
		
		self.hard_coded_model = DecisionTreeModel(decision_tree)

		return self.hard_coded_model