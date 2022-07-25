from tqdm import auto
import numpy as np


from classicalML.classification.models import decision_tree as dt


class RandomForestClassifier:

    def __init__(self,
                 estimators=10,
                 max_depth=5,
                 min_samples=5):

        self._estimators = [dt.DecisionTreeClassifier(max_depth,
                                                      min_samples)
                            for _ in range(estimators)]
        self._max_depth = max_depth
        self._min_samples = min_samples

    def __str__(self):

        return f'RandomForestClassifier' \
               f'(estimators={len(self._estimators)}' \
               f' max_depth={self._max_depth},' \
               f' min_samples={self._min_samples})'

    def _predict_for_sample(self,
                            x):

        return np.bincount([estimator.predict(x)
                            for estimator in self._estimators]).argmax()

    def fit(self,
            X_train,
            Y_train):

        for estimator in auto.tqdm(self._estimators):

            indices = np.random.choice(X_train.shape[0],
                                       size=X_train.shape[0],
                                       replace=True)
            estimator.fit(X_train[indices],
                          Y_train[indices])

    def predict(self,
                X):

        Y_predicted = []

        for x in X:

            y_predicted = self._predict_for_sample(x)
            Y_predicted.append(y_predicted)

        return Y_predicted
