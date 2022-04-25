import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn import preprocessing
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
import pandas as pd


# return dict of the data with different imputation methods
def imputation(data, columns, null_columns):
    data_with_mean = imputing_with_mean(data, columns, null_columns)
    data_with_median = imputing_with_median(data, columns, null_columns)
    data_with_most_frequent = imputing_with_most_frequent(data, null_columns)
    data_with_median_zero = imputing_with_zero(data, columns, null_columns)
    data_with_delete_null = imputing_with_delete_null(data, columns, null_columns)
    # data_with_knn = imputing_with_knn(data, columns)

    # todo knn and variance
    data_imputation = {'data_with_mean': data_with_mean, 'data_with_median': data_with_median,
                       'data_with_most_frequent': data_with_most_frequent,
                       'data_with_median_zero': data_with_median_zero, 'data_with_delete_null': data_with_delete_null}
    # print(data_with_knn['review_scores_rating'])
    return data_imputation


# instead of Nan value putting mean for numeric variables
def imputing_with_mean(data, columns, null_columns):
    data_with_mean_values = data.copy()
    for column in null_columns:
        if column in columns['numeric_variables']:
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            data_with_mean_values[column] = imputer.fit_transform(data_with_mean_values[column].values.reshape(-1, 1))
    return data_with_mean_values


# instead of Nan value putting median
def imputing_with_median(data, columns, null_columns):
    data_with_median_values = data.copy()
    for column in null_columns:
        if column in columns['numeric_variables']:
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            data_with_median_values[column] = imputer.fit_transform(data_with_median_values[column]
                                                                        .values.reshape(-1, 1))
    return data_with_median_values


# instead of Nan value putting most frequent value for all variables
def imputing_with_most_frequent(train, test, null_columns):
    train_with_most_frequent_values = train.copy()
    test_with_most_frequent_values = test.copy()
    for column in null_columns:
        val = train_with_most_frequent_values[column].value_counts().index[0]
        train_with_most_frequent_values[column] = train_with_most_frequent_values[column].fillna(val)
        test_with_most_frequent_values[column] = test_with_most_frequent_values[column].fillna(val)
    return train_with_most_frequent_values, test_with_most_frequent_values


# instead of Nan value putting zero values
def imputing_with_zero(data, columns, null_columns):
    data_with_zero_value = data.copy()
    for column in null_columns:
        if column in columns['numeric_variables']:
            imputer = SimpleImputer(missing_values=np.nan, fill_value=0, strategy='constant')
            data_with_zero_value[column] = imputer.fit_transform(data_with_zero_value[column].values.reshape(-1, 1))
    return data_with_zero_value


# removing rows with null
def imputing_with_delete_null(data, columns, null_columns):
    data_with_deleted_null = data.copy()
    data_with_deleted_null = data_with_deleted_null.dropna()
    return data_with_deleted_null


# using knn algorithm to imputation
def imputing_with_knn(data, columns):
    data_with_numric_column = data.copy()
    le = preprocessing.LabelEncoder()
    catgory_column_list = data_with_numric_column.select_dtypes(include=['object']).columns.tolist()
    for column in catgory_column_list:
        data_with_numric_column[column] = le.fit_transform(data_with_numric_column[column])
    data_with_knn_value = data_with_numric_column
    impute_knn = KNNImputer(n_neighbors=2)
    data_with_knn_value = pd.DataFrame(impute_knn.fit_transform(data_with_knn_value),
                                       columns=data_with_knn_value.columns)
    data_with_knn_value.head(20)
    return data_with_knn_value, data_with_numric_column
#
#
# # instead of Nan value putting covariance
# def imputing_with_covariance(data, columns, null_columns):
#     data_with_corr_value = data.copy()
#     for column in null_columns:
#         if column in columns['numeric_variables']:
#             impute_it = IterativeImputer()
#             data_with_corr_value = pd.DataFrame(impute_it.fit_transform(data_with_corr_value),
#                                                 columns=data_with_corr_value.columns)
#     data_with_corr_value.head()
#     return data_with_corr_value


