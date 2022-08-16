import numpy as np
import pandas as pd

from data_preparation.imputing_methods import imputing_with_most_frequent
from data_preparation.load_data import load, column_type, load_delete_null


# removing unnecessary columns from the dataset
def drop_columns(train, test, columns):
    for column in columns['column_to_drop']:
        train = train.drop(column, axis=1)
        test = test.drop(column, axis=1)
    return train, test


# covert TRUE, t to True and FALSE, f to False
def convert_boolean_category(train, test, columns):
    for column in columns['binary_variables']:
        train[column] = train[column].apply(lambda x: True if x == 'TRUE' or x == 't' else False)
        test[column] = test[column].apply(lambda x: True if x == 'TRUE' or x == 't' else False)
    return train, test


# covert TRUE, t to True and FALSE, f to False
def convert_percent_to_numeric(train, test):
    train['host_response_rate'] = train['host_response_rate'].apply(lambda x: x.strip('%')/100)
    test['host_response_rate'] = test['host_response_rate'].apply(lambda x: x.strip('%')/100)
    return train, test


def convert_to_float(train, test, columns):
    for column in columns['numeric_variables']:
        train[column] = train[column].astype(np.float)
        test[column] = test[column].astype(np.float)
    return train, test


def convert_date_cols_to_datetime(data, cols_list):
    for col in cols_list:
       data[col] = pd.to_datetime(data[col])
    return data


def load_and_clean_data(path):
    train, test, null_columns = load(path)
    columns = column_type()
    train, test = drop_columns(train, test, columns)
    train, test = convert_boolean_category(train, test, columns)
    train, test = convert_to_float(train, test, columns)
    # train_with_mean = imputation(train, columns, null_columns)
    # imputed_test = imputation(train, columns, null_columns)
    train, test = imputing_with_most_frequent(train, test, null_columns)

    return train, test


def clean_data_without_null(path):
    train, test = load_delete_null(path)
    columns = column_type()
    train, test = drop_columns(train, test, columns)
    train, test = convert_boolean_category(train, test, columns)
    train, test = convert_to_float(train, test, columns)


    return train, test
