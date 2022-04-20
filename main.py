from data_preparation.cleaning import load_and_clean_data
from data_preparation.wrangling_and_feature_engineering import *
from model.evaluate_results import evaluate_results
from model.model_analysis import model_analysis
from model.train import train_model


def run():
    train_with_mean, test = load_and_clean_data(path='dataset/train.csv')
    train, test = feature_engineering(train_with_mean, test)
    model = train_model(train)
    evaluate_results(model=model, test=test, train=train)


if __name__ == '__main__':
    run()
