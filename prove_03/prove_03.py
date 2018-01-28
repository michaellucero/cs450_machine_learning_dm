"""
Author: Michael Lucero
Assignment: Prove 3
Purpose: This program will take 3 data sets and practice preprocessing data by
doing, classification, normilization, and regression. Assignment still
inprogress with all these features not yet implemented.
"""


from myKNN import kNNClassifier
import data_preprocessing as dp

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier



def main():
    
    """
    Car DATA
    """

    # Set file path and headers for data (add directory)
    file_path_car = "/Users/michaellucero/Documents/BYUI/9 Winter 2018/CS 450 Machine Learning and Data Mining/cs450_machine_learning_dm/prove_03/car.data.txt"
    headers_car = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    
    # Read data from file (file is comma seperated)
    data_car = dp.read_data_from_file_car(headers_car, file_path_car)
    
    # handle non-numeric data and normilize values
    data_car = dp.handle_non_numeric_car(data_car)
    
    # Has No missing data

    # K-fold Cross Validation
   
    print(data_car)
    print()
    
    """
    Pima indians diabetes data
    """
    # Set file path and headers for data (add directory) 
    file_path_pima = "/Users/michaellucero/Documents/BYUI/9 Winter 2018/CS 450 Machine Learning and Data Mining/cs450_machine_learning_dm/prove_03/pima-indians-diabetes.data.txt"
    headers_pima = ["Number of times pregnant", "Plasma glucose concentration a 2 hours in an oral glucose tolerance test", "Diastolic blood pressure (mm Hg)", "Triceps skin fold thickness (mm)", "2-Hour serum insulin (mu U/ml)", "Body mass index (weight in kg/(height in m)^2)", "Diabetes pedigree function", "Age (years)", "Class variable (0 or 1)"]
    
    # There are only numeric values

    # Read from file (No Headers used and file is comma seperated)
    data_pima = dp.read_data_from_file_pima(None, file_path_pima)
    
    # handle missing data 0 encoded by replacing with mean
    data_pima = dp.handle_missing_data_zero(data_pima)

    # K-fold Cross Validation

    print(data_pima)
    print()


    """
    Auto MPG data 
    """
    # Set file path and headers for data (add directory)
    file_path_auto_mpg = "/Users/michaellucero/Documents/BYUI/9 Winter 2018/CS 450 Machine Learning and Data Mining/cs450_machine_learning_dm/prove_03/auto-mpg.data.txt"    
    headers_auto_mpg = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
   
   # Read from file (file is blank space seperated)
    data_auto_mpg = dp.read_data_from_file_auto_mpg(headers_auto_mpg, file_path_auto_mpg, True)

    # Already Numeric values and name is not need to convert

    # Handle missing values by removing from list missing values
    data_auto_mpg = dp.handle_missing_data_auto(data_auto_mpg)

    # K-fold Cross Validation


    print(data_auto_mpg)
    print()



    """
    PLACE HOLDER CODE FROM PONDER 2, NOT IMPLEMENTED
    """

    # # create model
    # classifier = kNNClassifier(k_neighbors=7)

    # # model is trained by saving neighbors points
    # model = classifier.fit(data_train, target_train)

    # print("Predicted Targets From my kNN Model:")
    # targets_predicted = model.predict(data_test)
    # print(targets_predicted)
    # print()

    # # determine accuracy
    # accuracy = accuracy_score(targets_predicted, target_test)
    # print("Accuracy In My Prediction: {:.2f}".format(accuracy))


    # classifier_2 = KNeighborsClassifier(n_neighbors=7)
    # model_2 = classifier_2.fit(data_train, target_train)
    # predictions_2 = model_2.predict(data_test)

    # print()
    # print("other implementation or sklearn.neighbors")
    # print(predictions_2)
    # print()

    # accuracy = accuracy_score(predictions_2, target_test)
    # print("Accuracy In sklearn.neighbors Implimentations Prediction: {:.2f}".format(accuracy))


if __name__ == '__main__':
    main()