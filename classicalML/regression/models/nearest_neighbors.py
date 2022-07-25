from tqdm import auto
import numpy as np


class KNearestNeighborsRegressor:

    def __init__(self,
                 k=5):

        self._k = k
        self._X_train = None
        self._Y_train = None

    def __str__(self):

        return f'KNearestNeighborsRegressor' \
               f'(n={self._k})'

    @staticmethod
    def _predict_for_sample(Y_subset):

        return np.mean(Y_subset)

    def fit(self,
            X_train,
            Y_train):

        self._X_train = X_train
        self._Y_train = Y_train

    def predict(self,
                X):

        Y_predicted = []

        for x in auto.tqdm(X):

            distances = np.array([np.sqrt(np.sum((x_train - x) ** 2))
                                  for x_train in self._X_train])
            indices = np.argsort(distances)[:self._k]
            y_predicted = self._predict_for_sample(self._Y_train[indices])
            Y_predicted.append(y_predicted)

        return np.array(Y_predicted)
