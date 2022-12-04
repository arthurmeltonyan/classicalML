from sklearn import datasets

from classicalML.dataset import cross_validation as cv
from classicalML.regression import metrics
from classicalML.regression.models import linear_model as lm
from classicalML.regression.models import decision_tree as dt
from classicalML.regression.models import nearest_neighbors as knn
from classicalML.regression.models import random_forest as rf
from classicalML.regression.models import support_vector_machine as svm


def check_regressor(X,
                    Y,
                    regressor):

    X_train, X_validation, Y_train, Y_validation = cv.split_train_validation_datasets(X,
                                                                                      Y,
                                                                                      validation_fraction=0.2)
    regressor.fit(X_train,
                  Y_train)
    Y_predicted = regressor.predict(X_validation)
    print(regressor)
    for metric in [metrics.root_mean_squared_error,
                   metrics.root_median_squared_error,
                   metrics.root_mean_squared_log_error,
                   metrics.root_median_squared_log_error,
                   metrics.mean_absolute_error,
                   metrics.median_absolute_error,
                   metrics.mean_absolute_log_error,
                   metrics.maximum_absolute_error]:
        metric_name = str(metric).split(' ')[1]
        metric_value = metric(Y_predicted, Y_validation)
        print(f'{metric_name} = {metric_value}')
    print()


def main():
    regression_dataset = datasets.make_regression(n_samples=10000,
                                                  n_features=10,
                                                  noise=20,
                                                  random_state=2020)
    check_regressor(*regression_dataset,
                    lm.LinearRegressor(alpha=0.001, epochs=1000))
    check_regressor(*regression_dataset,
                    dt.DecisionTreeRegressor(max_depth=8, min_samples=20))
    check_regressor(*regression_dataset,
                    knn.KNearestNeighborsRegressor(k=5))
    check_regressor(*regression_dataset,
                    rf.RandomForestRegressor(estimators=15, max_depth=8, min_samples=20))
    check_regressor(*regression_dataset,
                    svm.LinearSupportVectorRegressor(alpha=0.001, epochs=1000, parameter=1.0, epsilon=0.0))


if __name__ == '__main__':
    main()
