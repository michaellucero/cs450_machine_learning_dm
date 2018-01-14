"""
Author: Michael Lucero
Assignment: Ponder 1
Purpose: This program will use the iris data set
         to predict what type of iris it is using
         GaussianNB. Additionally this program will
         use a shell HardCodedClassifier that will 
         be populated with a model and intellegence.
    
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class HardCodedModel:
	"""
	Shell for future model
	"""
	def __init__(self):
		pass
	
	def predict(self, testing_data):

		predicted_targets = []
		
		for test in testing_data:
			predicted_targets.append(0)
		return predicted_targets

class HardCodedClassifier:
	"""
	Shell for future classifier
	"""	
	def __init__(self):
		pass
		
	def fit(self, training_data, training_targets):
		self.hard_coded_model = HardCodedModel()
		return self.hard_coded_model
		
		
	
	
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


# create model
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)

# make predictions
targets_predicted = model.predict(data_test)

print("30% of original")
print("Testing Data:")
print(data_test)
print()

print("Predicted Targets From Model Feed With Testing Data:")
print(targets_predicted)
print()

print("Actual Targets:")
print(target_test)
print()

accuracy = accuracy_score(targets_predicted, target_test)

print("Accuracy In Prediction: {:.2f}".format(accuracy))

"""
My Classifier
"""
print("My Hard Coded Classifer which will always return 0")
classifier = HardCodedClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)

print("30% of original")
print("Testing Data:")
print(data_test)
print()

print("Predicted Targets From Model Feed With Testing Data:")
print(targets_predicted)
print()

print("Actual Targets:")
print(target_test)
print()

accuracy = accuracy_score(targets_predicted, target_test)

print("Accuracy In Prediction: {:.2f}".format(accuracy))