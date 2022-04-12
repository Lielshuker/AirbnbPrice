import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path)
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7
    train = df[msk]
    test = df[~msk]
    return train, test


def column_type():
    binary_variables = ['cleaning_fee', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    categorical_variables = ['property_type', 'room_type', 'amenities', 'bed_type', 'cancellation_policy'
                             'city', 'description', 'first_review', 'host_since', 'last_review',
                             'name', 'neighbourhood', 'thumbnail_url']
    numeric_variables = ['id', 'log_price', 'accommodates', 'bathrooms', 'host_response_rate', 'latitude', 'longitude',
                         'number_of_reviews', 'review_scores_rating', 'zipcode', 'bedrooms', 'beds']
    return binary_variables, categorical_variables, numeric_variables
