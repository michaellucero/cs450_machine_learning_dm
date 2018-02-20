import pandas as pd
import numpy as np
from sklearn import preprocessing





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



def data_standardization(dataset):

    std_scale_data = preprocessing.StandardScaler().fit(dataset)
    df_std = std_scale_data.transform(dataset)
    print("standardized data:")
    print(df_std)
    print()
    return df_std

# CROSS_VALIDATION_PIMA
def cross_validation_pima(dataset):
    pass