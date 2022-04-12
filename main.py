from data_preparation.load_data import load


def run():
    train, test = load(path='dataset/train.csv')
    print(len(test)), print(len(train))


if __name__ == '__main__':
    run()
