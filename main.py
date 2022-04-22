from data_preparation.cleaning import load_and_clean_data
from data_preparation.load_data import column_type
from data_preparation.wrangling_and_feature_engineering import feature_engineering
from model.evaluate_results import evaluate_results
from model.train import train_model


def run():
    columns_dict = column_type()
    train_with_mean, test = load_and_clean_data(path='dataset/train.csv')
    train = train_with_mean['data_with_most_frequent']
    train, test = feature_engineering(train, test, columns_dict)
    model = train_model(train)
    evaluate_results(model=model, test=test, train=train)
    print('')


if __name__ == '__main__':
    run()
