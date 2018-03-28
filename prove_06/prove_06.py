"""
Author: Michael Lucero
Assignment: Prove 06
Purpose: 
"""

from myNeuralNet import NeuralNetClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import data_preprocessing as dp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

"""
MAIN
"""
def main():


    """
    iris data
    """
    #get dataset iris
#    iris = datasets.load_iris()
#    data, targets = iris.data, iris.target
    
    # print("Data:")
    # print(data)
    # print()
    # print("Targets:")
    # print(targets)
    # print()



    """
    Pima indians diabetes data
    """
    # Set file path and headers for data (add directory) 
    file_path_pima = "pima-indians-diabetes.data.txt"
    headers_pima = ["Number of times pregnant", "Plasma glucose concentration a 2 hours in an oral glucose tolerance test", "Diastolic blood pressure (mm Hg)", "Triceps skin fold thickness (mm)", "2-Hour serum insulin (mu U/ml)", "Body mass index (weight in kg/(height in m)^2)", "Diabetes pedigree function", "Age (years)", "Class variable (0 or 1)"]
    
    # There are only numeric values

    # Read from file (No Headers used and file is comma seperated)
    pima_df = dp.read_data_from_file_pima(None, file_path_pima)
    
    # handle missing data 0 encoded by replacing with mean
    pima_df = dp.handle_missing_data_zero(pima_df)

    print(pima_df)
    print()

    # convert to numpy array
    pima_np = pima_df.as_matrix()

    print(pima_np)

    #split into data and targets
    data, targets = pima_np[:, :-1], pima_np[:, -1]


    #test = np.array([[0.4, 0.6, 'A'], [0.2, 0.1, 'B'], [0.3, 0.9, 'A'], [0.3, 0.2, 'B'], [0.4, 0.7, 'A'], [0.1, 0.1, 'B'], [0.6, 0.9, 'A'], [0.3, 0.2, 'B']])
    #print("Test Data: ")
    #print(test)
    #print()

    #data, targets = test[:, :-1], test[:, -1]

    #standardize data
    data = dp.data_standardization(data)

    # print("Data:")
    # print(data)
    # print()
    # print("Targets:")
    # print(targets)
    # print()


    # randomize data into 70% training set and 30% testing set
    data_train, data_test, target_train, target_test = train_test_split(data, targets, train_size=0.70, test_size=0.30)

    # create model with parameters it specify layer shape. [layer node list], with the length of being
    #   the number of hidden layers
    classifier = NeuralNetClassifier([4], .1)
 
    # train model
    model = classifier.fit(data_train, target_train)
    
    # predict targets
    targets_predicted = model.predict(data_test)

    # Display results
    print("Predicted Targets From my neural network Model:")
    print(targets_predicted)
    print()

    #clf = MLPClassifier()
    #model_sk = clf.fit(data_train, target_train)
    #off_shelf_targets_predicted = model_sk.predict(data_test)
    #print("off shelf targets Predicted:")
    #print(off_shelf_targets_predicted)
    print()
    
    print("Actual Targets:")
    print(target_test)
    print()

    #print(data_train)
    #print(target_train)
    accuracy = accuracy_score(targets_predicted, target_test)
    print("Accuracy In My Prediction: {:.2f}".format(accuracy))
    
    #accuracy2 = accuracy_score(off_shelf_targets_predicted, target_test)
    #print("Accuracy for off the shelf Prediction: {:.2f}".format(accuracy2))

    MLPClassifier()
if __name__ == '__main__':
    main()





