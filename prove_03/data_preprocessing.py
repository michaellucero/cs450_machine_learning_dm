
import pandas as pd
import numpy as np
from sklearn import preprocessing

"""
Car Data preprocessing functions
"""
# READ_DATA_FROM_FILE_CAR
def read_data_from_file_car(headers, file_path):

    dataset = pd.read_csv(file_path, index_col=False, names=headers)
    # print(dataset.head(45))
    return dataset

# HANDLE_NON_NUMERIC_CAR
def handle_non_numeric_car(dataset):
    
    #print(dataset.dtypes)

    # basing on each attribute being part equal waiting 
    update_values = {"buying": {"vhigh": 4, "high": 3, "med":2 , "low": 1},
                     "maint": {"vhigh": 1, "high": 2, "med": 3, "low": 4},
                     "doors": {"2": 1, "3": 2, "4": 3, "5more": 4},
                     "persons": {"2": 1, "4": 2, "more": 3},
                     "lug_boot": {"small": 1, "med": 2, "big": 3},
                     "safety": {"low": 1, "med": 2, "high": 3}}

    dataset.replace(update_values, inplace=True)

    # print(dataset.head(30))

    # Normilize the data based on there individual
    minmax_scale  = preprocessing.MinMaxScaler().fit(dataset[["buying", "maint", "doors", "persons", "lug_boot", "safety"]])
    dataset = minmax_scale.transform(dataset[["buying", "maint", "doors", "persons", "lug_boot", "safety"]])
    
    # for data in dataset:
    #     print(data)

    return dataset


# CROSS_VALIDATION_CAR
def cross_validation_car(dataset):
    pass




"""
Pima indians diabetes data preprocessing functions
"""
# READ_DATA_FROM_FILE_PIMA
def read_data_from_file_pima(headers, file_path):
    dataset = pd.read_csv(file_path, index_col=False, header=headers, na_values="?")
    #print(dataset.head(45))
    return dataset


# HANDLE_MISSING_DATA_ZERO encode with '0' to only applicable columns and replace with the mean.
def handle_missing_data_zero(dataset):

    # not all are problems being '0'
    #print((dataset[[0, 1, 2, 3, 4, 5, 6, 7, 8]] == 0).sum())

    dataset[[1, 2, 3, 4, 5, 6]] = dataset[[1, 2, 3, 4, 5, 6]].replace(0, np.NaN)
    #print(dataset.isnull().sum())

    # Take the average of each column and insert in to NaN values
    dataset.fillna(dataset.mean(), inplace=True)

    #print(dataset.isnull().sum())

    return dataset




"""
Auto MPG data diabetes data preprocessing functions
"""
# READ_DATA_FROM_FILE_AUTO_MPG
def read_data_from_file_auto_mpg(headers, file_path, spaces):
    dataset = pd.read_table(file_path,  delim_whitespace=True, names=headers, na_values="?")
    #print(dataset.head(45))
    return dataset

# HANDLE_MISSING_DATA_AUTO remove them from dataset
def handle_missing_data_auto(dataset):
    #print(dataset.shape)
    dataset.dropna(inplace=True)
    #print(dataset.shape)
    return dataset


# CROSS_VALIDATION_AUTO_MPG
def cross_validation_auto_mpg(dataset):
    pass










# Scraped code saved just in case

# """
#     Will read data from file in either comma or spaces format and check for 
#     headers being provided
# """
# def read_data_from_file(headers, file_path, spaces):

#     if headers is None:
#         if spaces:
#             dataset = pd.read_table(file_path,  delim_whitespace=True, header=headers, na_values="?")
#         else:
#             dataset = pd.read_csv(file_path, index_col=False, header=headers, na_values="?")
#     else:
#         if spaces:
#             dataset = pd.read_table(file_path,  delim_whitespace=True, names=headers, na_values="?")
#         else:
#             dataset = pd.read_csv(file_path, index_col=False, names=headers, na_values="?")
    
#     #print(dataset.head(45))

#     return dataset



    # dataset_obj["buying"] = dataset_obj["buying"].astype('category')    
    # dataset_obj["maint"] = dataset_obj["maint"].astype('category')    
    # dataset_obj["doors"] = dataset_obj["doors"].astype('category')    
    # dataset_obj["persons"] = dataset_obj["persons"].astype('category')    
    # dataset_obj["lug_boot"] = dataset_obj["lug_boot"].astype('category')    
    # dataset_obj["safety"] = dataset_obj["safety"].astype('category')      

    # dataset_obj["buying"] = dataset_obj["buying"].cat.codes
    # dataset_obj["maint"] = dataset_obj["maint"].cat.codes
    # dataset_obj["doors"] = dataset_obj["doors"].cat.codes
    # dataset_obj["persons"] = dataset_obj["persons"].cat.codes
    # dataset_obj["lug_boot"] = dataset_obj["lug_boot"].cat.codes
    # dataset_obj["safety"] = dataset_obj["safety"].cat.codes
    #print(dataset.head(30))
    #print((dataset[[1,2,3,4,5]] == 0).sum())