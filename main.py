from data_preparation.cleaning import clean_data_without_null, load_and_clean_data

from data_preparation.load_data import column_type
from data_preparation.wrangling_and_feature_engineering import feature_engineering
from model.train import train_model


def run():
    train, test = load_and_clean_data(path='dataset/train.csv')
    train = train.reset_index()
    test = test.reset_index()


    columns_dict = column_type()
    train, test = feature_engineering(train, test, columns_dict)


    model = train_model(train)


if __name__ == '__main__':
    run()
