from data_preparation.cleaning import load_and_clean_data
from data_preparation.wrangling_and_feature_engineering import *


def run():
    load_and_clean_data(path='dataset/train.csv')

    # # Gil's binning and one hoe encoding full proccess
    # columns = None  # add dict (need to decide what variables are indeed categorial + add 'binned_variable' key
    # imputed_train, imputed_test = load_and_clean_data(path='dataset/train.csv')
    # train = imputed_train[4]  # delete null
    # test = imputed_test[4]
    # train = randomly_shuffle_training_data(train)
    # naive_binning(train, test, columns)
    # oh_train, oh_test = one_hot_enc(train, test, columns)
    # oh_train, oh_test = concat_binary_cols(train, test, oh_train, oh_test, columns)
    # oh_train, oh_test = equalize_columns(oh_train, oh_test)
    # return oh_train, oh_test


if __name__ == '__main__':
    run()
