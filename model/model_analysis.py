import shap
from matplotlib import pyplot as plt


def model_analysis(model, train_x):
    print("Model coefficients:\n")
    for i in range(len(train_x.columns)):
        print(train_x.columns[i], "=", model.coef_[i].round(4))


# def shap_cal(model, train_x, test_x, test_y, rel_res, prediction_test, residuals):
#     shap_sample = train_x.sample(25)
#     explainer = shap.Explainer(model.predict, shap_sample)
#     #shap_values = explainer(shap_sample)
#     #shap.plots.beeswarm(shap_values)
#     # test_shap_values = explainer(test_x)
#     print('test_x[rel_res == rel_res.max():', rel_res[rel_res == rel_res.max()])
#     print('prediction_test:', prediction_test[0])
#     #      print('test_x[rel_res == rel_res.max():', test_x[rel_res == rel_res.max()])
#     #      print('Get integer location of test_x:', test_x.index.get_loc(test_x[rel_res == rel_res.max()].iloc[:, 0]))
#     #      print('Get integer location of test_x:', test_y.loc[test_x[rel_res == rel_res.max()]], 'prediction_test:', prediction_test[20])
#     # shap.plots.waterfall(test_shap_values[0])
#     print('residuals values bigger then 5000:', len(residuals[residuals > 50000]))
#     bad_examples = test_x.loc[residuals[residuals > 50000].index]
#     bad_examples_shap_values = explainer(bad_examples)
#     shap.plots.beeswarm(bad_examples_shap_values)
#     over_estimates = test_x.loc[residuals[residuals>50000].index]
#     good_estimates = test_x.loc[rel_res[rel_res < 0.05].index]
#     fig, ax = plt.subplots(figsize=(10, 4))
#     good_estimates.GrLivArea.hist(ax=ax, color='blue')
#     over_estimates.GrLivArea.hist(ax=ax, color='red')






