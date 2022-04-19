import pandas as pd
import data_preparation.cleaning
from datetime import datetime
import numpy as np

from data_preparation.load_data import convert_present_to_float


def collect_amenities(data, columns_dict):
    chars_to_remove = '"{}'
    amenities_set = set()
    for line in data['amenities']:
        amenities = line.split(',')
        for obj in amenities:
            for char in chars_to_remove:
                obj = obj.replace(char, "")
            if "translation missing" not in obj and obj != '':
                amenities_set.add(obj)
    for amenity in amenities_set:
        columns_dict['binary_variables'].append(amenity)
    return list(amenities_set)


def create_amenities_array(amenities_list, data):
    amenities_array = []
    for index, row in data.iterrows():
        array = np.zeros(shape=(len(amenities_list)))
        row_amen = data['amenities'][index].split(',')
        for amen in row_amen:
            item = amen.replace('"', '').replace('}', '').replace('{', '')
            if item in amenities_list:
                res = amenities_list.index(item)
                array[res] = 1
        amenities_array.append(array.tolist())

    amenities_df = pd.DataFrame(amenities_array, columns=amenities_list)
    return amenities_df


# converting amenities column to binary columns and updating columns_dict
def create_amenities_cols(data, amenities_set):
    amenities_array = create_amenities_array(amenities_set, data)

    data = data.drop(['amenities'], axis=1)
    data = pd.concat([data, amenities_array], axis=1)

    return data


def handle_amenities(train, test, columns_dict):
    amenities_list = collect_amenities(train, columns_dict)
    train = create_amenities_cols(train, amenities_list)
    test = create_amenities_cols(test, amenities_list)
    columns_dict['categorical_variables'].remove('amenities')

    return train, test, columns_dict


def randomly_shuffle_training_data(train):
    return train.sample(frac=1)


def handle_neighbourhood(train, test):
    top_neighbourhoods = train['neighbourhood'].value_counts().head(50).keys()
    for index, row in train.iterrows():
        if row['neighbourhood'] not in top_neighbourhoods:
            train.at[index,'neighbourhood'] = 'other'
    for index, row in test.iterrows():
        if row['neighbourhood'] not in top_neighbourhoods:
            test.at[index,'neighbourhood'] = 'other'

    return train, test


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


def concat_numeric_cols(train, test, oh_train, oh_test, columns):
    for col in columns['numeric_variables']:
        if col != 'log_price':
            oh_train = pd.concat([oh_train, train[col]], axis=1)

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


def feature_engineering(train, test, columns_dict):
    train, test, columns_dict = handle_amenities(train, test, columns_dict)
    train, test = handle_neighbourhood(train, test)
    oh_train, oh_test = one_hot_enc(train, test, columns_dict)
    oh_train, oh_test = concat_binary_cols(train, test, oh_train, oh_test, columns_dict)
    oh_train, oh_test = concat_numeric_cols(train, test, oh_train, oh_test, columns_dict)
    oh_train, oh_test = equalize_columns(oh_train, oh_test)
    return oh_train, oh_test

