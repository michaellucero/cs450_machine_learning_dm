
import pandas as pd
import numpy as np
from sklearn import preprocessing

"""
lenses Data preprocessing functions
"""
# READ_DATA_FROM_lenses
def read_data_from_lenses(file_path, headers):

    dataset = pd.read_csv(file_path, delim_whitespace=True, names=headers)

    return dataset

#CONVERST_NUMMERIC_TO_ATTRIBUTE_TEXT
def convert_numeric_to_attribute_text(dataset):
    
    # update values to categorize
    update_values = {"age of the patient": {1: "young", 2: "pre-presbyopic", 3: "presbyopic"},
                     "maispectacle prescription": {1: "myope", 2: "hypermetrope"},
                     "astigmatic": {1: "yes", 2: "no"},
                     "tear production rate": {1 :"reduced", 2: "normal"},
                     "diagonsis": {1: "the patient should be fitted with hard contact lenses", 2: "the patient should be fitted with soft contact lenses", 3: "the patient should not be fitted with contact lenses"}}

    dataset.replace(update_values, inplace=True)
    print(dataset.head(45))

# READ_DATA_FROM_lenses
def read_data_from_credit(file_path, headers):

    dataset = pd.read_csv(file_path, delim_whitespace=True, names=headers)

    return dataset

#CONVERST_NUMMERIC_TO_ATTRIBUTE_TEXT
def convert_attribute_text_to_numeric(dataset):
    
    # update values to categorize
    update_values = {"credit score": {"Good": 1, "Average": 2, "Low": 3},
                     "income": {"High": 1,"Low": 2},
                     "collateral": {"Good": 1, "Poor": 2},
                     "should loan": {"Yes": 1, "No": 2}}
                     

    dataset.replace(update_values, inplace=True)
    print(dataset.head(5))
