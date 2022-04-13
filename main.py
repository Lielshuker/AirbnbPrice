from data_preparation.cleaning import load_and_clean_data


def run():
    load_and_clean_data(path='dataset/train.csv')


if __name__ == '__main__':
    run()
