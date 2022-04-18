import numpy as np
import pandas as pd


def convert_present_to_float(x):
    if x is '':
        return 0.0
    return float(x.strip('%'))/100


# creating dictionary with the names of each column and splitting the columns to binary, categorical and numeric
def column_type():
    binary_variables = ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    categorical_variables = ['property_type', 'room_type', 'amenities', 'bed_type', 'cancellation_policy', 'city',
                             'neighbourhood']
                            # name, thumbnail_url
    numeric_variables = ['log_price', 'accommodates', 'bathrooms', 'host_response_rate', 'latitude', 'longitude',
                         'number_of_reviews', 'review_scores_rating', 'bedrooms', 'beds', 'first_review',
                         'host_since', 'last_review']  # id, zipcode
    column_to_drop = ['id', 'name', 'thumbnail_url', 'zipcode', 'description']

    columns = {'binary_variables': binary_variables, 'categorical_variables': categorical_variables,
               'numeric_variables': numeric_variables, 'column_to_drop': column_to_drop}
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
    df = pd.read_csv(path, converters={'host_response_rate': convert_present_to_float})
    null_columns = check_null_values(df)
    df['split'] = np.random.randn(df.shape[0], 1)
    # todo
    msk = np.random.rand(len(df)) <= 1
    train = df[msk]
    test = df[~msk]

    return train, test, null_columns
