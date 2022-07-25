from tqdm import auto
import numpy as np


class LinearSupportVectorClassifier:

    def __init__(self,
                 alpha=0.001,
                 epochs=1000,
                 parameter=1.0,
                 report=False):

        self._bias = None
        self._weights = None
        self._alpha = alpha
        self._epochs = epochs
        self._parameter = parameter
        self._report = report

    def __str__(self):

        return f'LinearSupportVectorClassifier' \
               f'(alpha={self._alpha}, ' \
               f' epochs={self._epochs}, ' \
               f' parameter={self._parameter})'

    def _hypothesis(self,
                    x):

        return np.dot(x, self._weights) + self._bias

    def _cost(self,
              X_train,
              Y_train):

        n, k = X_train.shape
        first_term = 1 / 2 * np.sum(self._weights ** 2)
        second_term = self._parameter / n * np.sum([np.maximum(0, 1 - y * self._hypothesis(x))
                                                    for x, y in zip(X_train, Y_train)])

        return first_term + second_term

    def fit(self,
            X_train,
            Y_train):

        n, k = X_train.shape
        self._bias = 0
        self._weights = np.zeros(k)
        indices = np.random.choice(X_train.shape[0],
                                   size=np.minimum(self._epochs, X_train.shape[0]),
                                   replace=False)

        for epoch, index in enumerate(auto.tqdm(indices)):

            x = X_train[index]
            y = Y_train[index]

            if y * self._hypothesis(x) >= 1:

                gradient_bias = 0
                gradient_weights = self._weights

            else:

                gradient_bias = - self._parameter * y
                gradient_weights = self._weights - self._parameter * y * x

            self._bias -= self._alpha * gradient_bias
            self._weights -= self._alpha * gradient_weights

            if int(epoch * 100 / self._epochs) > int((epoch - 1) * 100 / self._epochs) and self._report:

                auto.tqdm.write(f'{epoch:>10}: {str(self._cost(X_train, Y_train))}')

    def predict(self,
                X):

        return np.sign(self._hypothesis(X))
