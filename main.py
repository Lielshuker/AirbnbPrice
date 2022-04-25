from data_preparation.cleaning import load_and_clean_data, clean_data_without_null
from data_preparation.load_data import column_type, load_delete_null
from data_preparation.wrangling_and_feature_engineering import feature_engineering, collect_amenities, create_amenities_cols, \
    handle_neighbourhood
from model.evaluate_results import evaluate_results
from model.train import train_model


def run():
    train, test = clean_data_without_null(path='dataset/train.csv')
    train = train.reset_index()
    test = test.reset_index()

    columns_dict = column_type()
    train, test = feature_engineering(train, test, columns_dict)


    model = train_model(train)
    evaluate_results(model=model, test=test, train=train)


if __name__ == '__main__':
    run()
