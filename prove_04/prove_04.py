"""
Author: Michael Lucero
Assignment: Prove 4
Purpose: 
"""

from myDecisionTree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import data_preprocessing as dp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

"""
MAIN
"""
def main():
    
    # get dataset iris
    iris = datasets.load_iris()

    # get dataset lenses
    file_path_lenses = ""
    headers_lenses = ["age of the patient", "maispectacle prescription", "astigmatic", "tear production rate", "diagonsis"]
    lenses_df = dp.read_data_from_lenses(file_path_lenses, headers_lenses)
    #dp.convert_numeric_to_attribute_text(lenses_df)
    lenses_np = lenses_df.as_matrix()

    # replace number with attribute
    

    # test data from class activity
    file_path_credit = ""
    headers_credit = ["credit score", "income", "collateral", "should loan"]
    credit_df = dp.read_data_from_credit(file_path_credit, headers_credit)
    dp.convert_attribute_text_to_numeric(credit_df)
    credit_np = credit_df.as_matrix()

    #data, targets = iris.data, iris.target
    data, targets = credit_np[:, :-1], credit_np[:, -1]
    #data, targets = lenses_np[:, :-1], lenses_np[:, -1]
    print()
    print("Data:")
    print(data)
    print()
    print("Targets:")
    print(targets)
    print()


    # randomize data into 70% training set and 30% testing set
    #data_train, data_test, target_train, target_test = train_test_split(credit_data, credit_targets, train_size=0.70, test_size=0.30)
    #data_train, data_test, target_train, target_test = train_test_split(data, targets, train_size=0.70, test_size=0.30)

    # create model
    classifier = DecisionTreeClassifier()

    # train model
    model = classifier.fit(data, targets)

#     # predict targets
#     targets_predicted = model.predict(data_test)

#     # Display results
#     #print("Predicted Targets From my ID3 Model:")
#    # print(targets_predicted)
#     print()

  #iris = datasets.load_iris()
    iris = datasets.load_iris()
    #print(iris.data)
    # randomize data into 70% training set and 30% testing set
    #data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, train_size=0.70, test_size=0.30)


    clf = tree.DecisionTreeClassifier()
    model_sk = clf.fit(data, targets)
    #print(model_sk.tree_.value)
    targets_predicted = model_sk.predict(data)
    # print("targets Predicted:")
    # print(targets_predicted)
    # print("targets:")
    # print(target_test) 

   # accuracy = accuracy_score(targets_predicted, target_test)
    #print("Accuracy In My Prediction: {:.2f}".format(accuracy))

if __name__ == '__main__':
    main()





