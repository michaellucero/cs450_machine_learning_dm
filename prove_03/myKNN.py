import numpy as np
from collections import Counter

class kNNModel:
	"""
	k Nearest Neighbor
	"""
	def __init__(self, k_neighbors, data_train, targets_train):
		# considered trained when data is saved in class
		self.k_neighbors = k_neighbors
		self.data_train = data_train
		self.targets_train = targets_train
	
	# kNN
	def predict(self, data_test):
		
		predicted_targets = []
		distances = np.array([])
		
		# calculate distance from data by distance formula
		for test_row in data_test:
			for train_row in self.data_train:
				distances = np.append(distances, ((test_row[0] - train_row[0]) ** 2) + \
					((test_row[1] - train_row[1]) ** 2) + \
					((test_row[2] - train_row[2]) ** 2) + \
					((test_row[3] - train_row[3]) ** 2))
	
			# get the lowest k values in the array by argsort and slicing
			k_smallest_indexes = (distances.argsort()[0:self.k_neighbors])

			# reset array
			distances = np.array([])	

			# determine the predicted target based on which of the k closes targets
			# 	comes up more often	
			pre_predicted_targets = []
			for k_smallest_index in k_smallest_indexes:
				pre_predicted_targets.append(self.targets_train[k_smallest_index])
						
			# Counter class produces a list of the count of each value
			pre_predicted_targets_value_counts = Counter(pre_predicted_targets)
			# Get the most common value in the list
			predicted_targets_dictionary = pre_predicted_targets_value_counts.most_common(1)
			# access the most common value
			predicted_targets.append(predicted_targets_dictionary[0][0])
		
		return predicted_targets

class kNNClassifier:
	"""
	Shell for future classifier
	"""	
	def __init__(self, k_neighbors):
		self.k_neighbors = k_neighbors
		
	def fit(self, data_train, targets_train):
		self.model = kNNModel(self.k_neighbors, data_train, targets_train)
		return self.model
			
	