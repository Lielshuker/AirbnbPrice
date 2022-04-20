from sklearn import linear_model


def train_model(train_dataset):
    linear_regression = linear_model.LinearRegression()
    # separate labels from data=
    train_class = train_dataset['log_price']
    oh_train_data = train_dataset.drop('log_price', axis=1)
    # train the model:
    linear_regression.fit(oh_train_data, train_class)
    print(linear_regression)
    return linear_regression
