import numpy as np
import pandas as pd

from data_preparation.imputing_methods import imputing_with_delete_null


def convert_present_to_float(x):
    if x is '':
        return 0.0
    return float(x.strip('%'))/100


# creating dictionary with the names of each column and splitting the columns to binary, categorical and numeric
def column_type():
    binary_variables = ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    categorical_variables = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city',
                             'neighbourhood', 'amenities'] # name, thumbnail_url
    numeric_variables = ['log_price', 'accommodates', 'bathrooms', 'host_response_rate', 'latitude', 'longitude',
                         'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds']  # id, zipcode
    column_to_drop = ['id', 'name', 'thumbnail_url', 'zipcode', 'description', 'first_review',
                         'host_since', 'last_review']

    columns = {'binary_variables': binary_variables, 'categorical_variables': categorical_variables,
               'numeric_variables': numeric_variables, 'column_to_drop': column_to_drop, 'binned_variables': []}

    return columns


def check_null_values(df):
    null_columns = []
    columns = column_type()
    for column in columns['binary_variables'] + columns['categorical_variables'] + columns['numeric_variables']:
        if df[column].isnull().sum():
            null_columns.append(column)
    return null_columns


# loading the dataset and split it to train - 80% and test - 20%
def load(path):
    # todo cancel 100 rows
    df = pd.read_csv(path, converters={'host_response_rate': convert_present_to_float})#, nrows=10000)
    null_columns = check_null_values(df)
    df['split'] = np.random.randn(df.shape[0], 1)
    # todo
    msk = np.random.rand(len(df)) <= 0.8
    train = df[msk]
    test = df[~msk]
    return train, test, null_columns


# loading the dataset, remove rows with null and split it to train - 80% and test - 20%
def load_delete_null(path):
    # todo cancel 100 rows
    df = pd.read_csv(path, converters={'host_response_rate': convert_present_to_float})#, nrows=10000)
    null_columns = check_null_values(df)
    columns = column_type()
    df = imputing_with_delete_null(df, columns, null_columns)
    df['split'] = np.random.randn(df.shape[0], 1)
    # todo
    msk = np.random.rand(len(df)) <= 0.8
    train = df[msk]
    test = df[~msk]
    return train, test
