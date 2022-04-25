import numpy as np
import seaborn as sns

from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, r2_score, mean_absolute_percentage_error, \
    mean_absolute_error, mean_squared_error
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from model.model_analysis import model_analysis #, shap_cal


def evaluate_results(test,train, model):
    train_y = train['log_price'].values
    train_x = train.drop('log_price', axis=1)
    test_y = test['log_price'].values
    test_x = test.drop('log_price', axis=1)

    prediction_test = model.predict(test_x)
    prediction_train = model.predict(train_x)

    plot_evaluation(
        model=model,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        prediction_test=prediction_test
    )


def plot_evaluation(model, train_x, train_y, test_x, test_y, prediction_test):
    #show_metrics(prediction_test, prediction_train, test_y, train_y)
    r2 = r2_score(test_y, prediction_test)
    print('r2 score:', r2)

    print("Mean Absolute Perc Error (Σ(|y-pred|/y)/n):",
          "{:,.3f}".format(mean_absolute_percentage_error(test_y, prediction_test)))
    print("Mean Absolute Error (Σ|y-pred|/n):", "{:,.3f}".format(mean_absolute_error(test_y, prediction_test)))
    print("Root Mean Squared Error (sqrt(Σ(y-pred)^2/n)):", "{:,.3f}".
          format(np.sqrt(mean_squared_error(test_y, prediction_test))))

    ## residuals
    residuals = test_y - prediction_test
    max_error = max(prediction_test) if abs(max(residuals)) > abs(min(residuals)) else min(residuals)
    max_idx = list(residuals).index(max(residuals)) if abs(max(residuals)) > abs(min(residuals)) else list(
        residuals).index(min(residuals))
    max_true, max_pred = test_y[max_idx], prediction_test[max_idx]
    print("Max Error:", "{:,.0f}".format(max_error))

    # fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=prediction_test, y=test_y)
    sns.lineplot(x=prediction_test, y=prediction_test, color='black')
    plt.title('true values against the predicted values')
    plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    sns.scatterplot(x=prediction_test, y=residuals, ax=ax[0])
    sns.lineplot(x=prediction_test, y=0, ax=ax[0], color='black')
    ax[0].set_title("Residuals (Abs)")
    sns.scatterplot(x=prediction_test, y=residuals / test_y, ax=ax[1])
    sns.lineplot(x=prediction_test, y=0, ax=ax[1], color='black')
    ax[1].set_title("Residuals (%)")
    plt.show()

    rel_res = residuals / test_y

    rel_res = abs(rel_res)
    print(len(rel_res[rel_res < 0.05]) / len(rel_res))
    print(len(rel_res[rel_res > 0.2]) / len(rel_res))

    model_analysis(model, train_x)
    # todo - shap
    # shap_cal(model, train_x, test_x, test_y, rel_res, prediction_test, residuals)

    # https://www.kaggle.com/code/mohamedmokhtar7/airbnb-eda-and-regression#kln-346
    sns.regplot(x=test_y, y=prediction_test, fit_reg=False)
    plt.title('Prediction and real')
    plt.show()

    sns.distplot(test_y - prediction_test, bins=50)
    plt.title('Error variance')
    plt.show()






