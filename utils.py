import numpy as np
import pandas as pd
import datetime
import shutil
import requests


# impute missing values
def impute_missing_values(train,test):
    null_columns = []
    for column in train.columns:
        if column != 'thumbnail_url':
            if train[column].isnull().sum():
                null_columns.append(column)
            if test[column].isnull().sum():
                null_columns.append(column)

    train_imputed = train.copy()
    test_imputed = test.copy()
    for column in null_columns:
        if train[column].dtype == 'float64':
            val = float(round(train_imputed[column].mean()))
            train_imputed[column] = train_imputed[column].fillna(val)
            test_imputed[column] = test_imputed[column].fillna(val)
        else:
            val = train_imputed[column].value_counts().index[0]
            train_imputed[column] = train_imputed[column].fillna(val)
            test_imputed[column] = test_imputed[column].fillna(val)

    train = train_imputed
    test = test_imputed

    return train, test


# amenities
def collect_amenities(data, binary_variables):
    chars_to_remove = '"{}'
    amenities_list = []
    amenities_dict = dict()
    for line in data['amenities']:
        amenities = line.split(',')
        for obj in amenities:
            for char in chars_to_remove:
                obj = obj.replace(char, "")
            if "translation missing" not in obj and obj != '':
                if obj not in amenities_dict:
                    amenities_dict[obj] = 1
                else:
                    amenities_dict[obj] += 1
    amenities_dict = sorted(amenities_dict, key=amenities_dict.get, reverse=True)
    for amenity in amenities_dict[:50]:
        amenities_list.append(amenity)
        binary_variables.append(amenity)
    return amenities_list


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


def create_amenities_cols(data, amenities_set):
    amenities_array = create_amenities_array(amenities_set, data)

    data = data.drop(['amenities'], axis=1)
    data = pd.concat([data, amenities_array], axis=1)

    return data


def convert_amenities_col(train, test, binary_variables):
    amenities_list = collect_amenities(train, binary_variables)
    train = create_amenities_cols(train, amenities_list)
    test = create_amenities_cols(test, amenities_list)

    return train, test


# convert date values to a numeric values
def convert_date_to_years_since(data, dates_variables):
    for var in dates_variables:

        data[var] = pd.to_datetime(data[var], format='%Y-%m-%d')
        data[var] = datetime.datetime.now() - data[var]
        data[var] = data[var].apply(lambda x: x.days)
        data[var] = data[var].apply(lambda x: x/365)
        data[var] = pd.to_numeric(data[var])

    return data



