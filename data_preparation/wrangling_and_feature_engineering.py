import pandas as pd
from cleaning import load_and_clean_data
from datetime import datetime
import numpy as np


def collect_amenities(data):
    chars_to_remove = '"{}'
    amenities_set = set()
    for line in data['amenities']:
        amenities = line.split(',')
        for obj in amenities:
            for char in chars_to_remove:
                obj = obj.replace(char, "")
            if "translation missing" not in obj and obj != '':
                amenities_set.add(obj)
    return amenities_set


def create_amenities_array(amenities, data):
    amenities_list = list(amenities)
    amenities_array = []
    for row in range(0, data.shape[0]):
        array = np.zeros(shape=(len(amenities)))
        row_amen = data['amenities'][row].split(',')
        for amen in row_amen:
            item = amen.replace('"', '').replace('}', '').replace('{', '')
            if "translation missing" not in item and item !='':
                res = amenities_list.index(item)
                array[res] = 1
        amenities_array.append(array.tolist())

    amenities_df = pd.DataFrame(amenities_array, columns=amenities_list)
    return amenities_df


# converting amenities column to binary columns and updating columns_dict
def handle_amenities(data, columns_dict):
    amenities_set = collect_amenities(data)
    amenities_array = create_amenities_array(amenities_set, data)

    data = data.drop(['amenities'], axis=1)
    data = pd.concat([data, amenities_array], axis=1)

    for amenity in amenities_set:
        columns_dict['binary_variables'].append(amenity)

    return data, columns_dict


def randomly_shuffle_training_data(train):
    return train.sample(frac=1)


def convert_date_cols_to_datetime(cols_list):
    for col in cols_list:
       train[col] = pd.to_datetime(train[col])


def col_binning(col, bin_num):
    # define min and max values:
    minval = col.min()
    maxval = col.max()

    # binning using cut function of pandas
    colBin = pd.cut(col,bin_num, include_lowest=True, precision=0)

    # rename the categories after the interval
    colBin=colBin.cat.rename_categories([f'{colBin.left}_{colBin.right}' for colBin in colBin.cat.categories])

    return colBin


def naive_binning(train, test, columns, bin_num=20):
    for col in columns['numeric_variables']:
        if col != 'log_price':
            train[col+'_binned'] = col_binning(train[col], bin_num)
            test[col+'_binned'] = col_binning(test[col], bin_num)
            columns['binned_variables'].append(col+'_binned')


# Encode all binned numeric columns and categorical columns with oneHot
def one_hot_enc(train, test, columns):

    # we leave the target feature as is
    oh_train = train['log_price']
    oh_test = test['log_price']

    # now adding the one hot encoded data
    for variable in columns['binned_variables']+columns['categorical_variables']:
        onehot_train_col = pd.get_dummies(train[variable], prefix=variable)
        oh_train = pd.concat([oh_train, onehot_train_col], axis=1)

        onehot_test_col = pd.get_dummies(test[variable], prefix=variable)
        oh_test = pd.concat([oh_test, onehot_test_col], axis=1)
    return oh_train, oh_test


# adding binary columns to one hot encoded dataset
def concat_binary_cols(train, test, oh_train, oh_test, columns):
    for col in columns['binary_variables']:
        train[col] = train[col].replace(True, 1)
        train[col] =train[col].replace(False, 0)
        oh_train = pd.concat([oh_train, train[col]], axis=1)

        test[col] = test[col].replace(True, 1)
        test[col] = test[col].replace(False, 0)
        oh_test = pd.concat([oh_test, test[col]], axis=1)

    return oh_train, oh_test


def equalize_columns(train, test):
    add_to_test = list(set(train.columns) - set(test.columns))
    add_to_train = list(set(test.columns) - set(train.columns))
    for col in add_to_train:
        train[col] = 0
    for col in add_to_test:
        test[col] = 0

    test = test[train.columns]

    return train, test
