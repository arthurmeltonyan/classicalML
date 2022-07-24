from tqdm import auto
import numpy as np


class KNearestNeighborsClassifier:

    def __init__(self,
                 k=5):

        self._k = k
        self._X_train = None
        self._Y_train = None

    def __str__(self):

        return f'KNearestNeighborsClassifier' \
               f'(n={self._k})'

    @staticmethod
    def _value(Y_subset):

        return np.bincount(Y_subset).argmax()

    def fit(self,
            X_train,
            Y_train):

        self._X_train = X_train
        self._Y_train = Y_train

    def predict(self,
                X_validation):

        Y_predicted = []

        for x_validation in auto.tqdm(X_validation):

            distances = np.array([np.sqrt(np.sum((x - x_validation) ** 2))
                                  for x in self._X_train])
            indices = np.argsort(distances)[:self._k]
            y_predicted = self._value(self._Y_train[indices])
            Y_predicted.append(y_predicted)

        return np.array(Y_predicted)
