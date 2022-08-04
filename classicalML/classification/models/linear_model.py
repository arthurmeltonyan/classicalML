from tqdm import auto
import numpy as np


class LinearClassifier:

    def __init__(self,
                 alpha=0.001,
                 epochs=1000,
                 report=False):

        self._bias = None
        self._weights = None
        self._alpha = alpha
        self._epochs = epochs
        self._report = report

    def __str__(self):

        return f'LinearClassifier' \
               f'(alpha={self._alpha},' \
               f' epochs={self._epochs})'

    def _hypothesis(self,
                    X):

        return 1 / (1 + np.exp(-(np.dot(X, self._weights) + self._bias)))

    def _cost(self,
              X_train,
              Y_train):

        n, m = X_train.shape
        first_term = np.dot(Y_train, np.log(self._hypothesis(X_train)))
        second_term = np.dot(1 - Y_train, np.log(1 - self._hypothesis(X_train)))

        return -1 / n * np.sum(first_term + second_term)

    def fit(self,
            X_train,
            Y_train):

        n, m = X_train.shape
        self._bias = 0
        self._weights = np.zeros(m)
        indices = np.random.choice(X_train.shape[0],
                                   size=np.minimum(self._epochs, X_train.shape[0]),
                                   replace=False)

        for epoch, index in enumerate(auto.tqdm(indices)):

            x = X_train[index]
            y = Y_train[index]
            error = self._hypothesis(x) - y
            gradient_bias = error
            gradient_weights = error * x
            self._bias -= self._alpha * gradient_bias
            self._weights -= self._alpha * gradient_weights

            if self._report and int(epoch * 100 / self._epochs) > int((epoch - 1) * 100 / self._epochs):
                auto.tqdm.write(f'{epoch:>10}: {str(self._cost(X_train, Y_train))}')

    def predict(self,
                X,
                threshold=0.5):

        return (self._hypothesis(X) > threshold).astype(int)
