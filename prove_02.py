"""
Author: Michael Lucero
Assignment: Prove 2
Purpose: This program implements the k nearest neighbor algorthm
with the iris data training set.
    
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

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
			
	

iris = datasets.load_iris()

# Show the data (the attributes of each instance)
print("Iris Data:")
print(iris.data)
print()

# Show the target values (in numeric format) of each instance
print("Iris Target:")
print(iris.target)
print()

# Show the actual target names that correspond to each number
print("Iris Target Names")
print(iris.target_names)
print()

# combines all attributes together into one array for verifying
data_target = []
for data, target in zip(iris.data, iris.target):
	data_target.append((data, target, iris.target_names[target]))	

#print(data_target)
print()


# randomize data into 70% training set and 30% testing set
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, train_size=0.70, test_size=0.30)

print("70% of original")
print("Training Data:")
print(data_train)
print()
print("Training Target:")
print(target_train)
print()
print("size of training data")
print(len(data_train))
print("30% of original")
print("Testing Data:")
print(data_test)
print()

print("Actual Targets:")
print(target_test)
print()

# create model
classifier = kNNClassifier(k_neighbors=7)

# model is trained by saving neighbors points
model = classifier.fit(data_train, target_train)

print("Predicted Targets From my kNN Model:")
targets_predicted = model.predict(data_test)
print(targets_predicted)
print()

# determine accuracy
accuracy = accuracy_score(targets_predicted, target_test)
print("Accuracy In My Prediction: {:.2f}".format(accuracy))


classifier_2 = KNeighborsClassifier(n_neighbors=7)
model_2 = classifier_2.fit(data_train, target_train)
predictions_2 = model_2.predict(data_test)

print()
print("other implementation or sklearn.neighbors")
print(predictions_2)
print()

accuracy = accuracy_score(predictions_2, target_test)
print("Accuracy In sklearn.neighbors Implimentations Prediction: {:.2f}".format(accuracy))
