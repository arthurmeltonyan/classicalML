from tqdm import auto
import numpy as np


import decision_tree as dt


class RandomForestRegressor:

    def __init__(self,
                 estimators=10,
                 max_depth=5,
                 min_samples=5):

        self._estimators = [dt.DecisionTreeRegressor(max_depth,
                                                     min_samples)
                            for _ in range(estimators)]
        self._max_depth = max_depth
        self._min_samples = min_samples

    def __str__(self):

        return f'RandomForestRegressor' \
               f'(estimators={len(self._estimators)}' \
               f' max_depth={self._max_depth},' \
               f' min_samples={self._min_samples})'

    def fit(self,
            X_train,
            Y_train):

        for estimator in auto.tqdm(self._estimators):

            indices = np.random.choice(X_train.shape[0],
                                       X_train.shape[0],
                                       replace=True)
            estimator.fit(X_train[indices],
                          Y_train[indices])

    def predict(self,
                X):

        return np.mean([estimator.predict(X)
                        for estimator in self._estimators], axis=0)
