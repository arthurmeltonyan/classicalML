import numpy as np


class NaiveBayesClassifier:

    def __init__(self):

        self._y_classes = None
        self._y_means = None
        self._y_variances = None
        self._y_prior_probabilities = None

    def __str__(self):

        return 'NaiveBayesClassifier()'

    def fit(self,
            X_train,
            Y_train):

        n, k = X_train.shape
        self._y_classes = np.unique(Y_train)
        m = self._y_classes.shape[0]
        self._y_means = np.zeros((m, k))
        self._y_variances = np.zeros((m, k))
        self._y_prior_probabilities = np.zeros(m)

        for y_class in self._y_classes:
            indices = np.argwhere(Y_train == y_class)
            self._y_means[y_class, :] = np.mean(X_train[indices], axis=0)
            self._y_variances[y_class, :] = np.var(X_train[indices], axis=0)
            self._y_prior_probabilities[y_class] = indices.shape[0] / Y_train.shape[0]

    def predict(self,
                X):

        Y_predicted = []

        for x in X:

            y_log_posterior_probabilities = []
            for y_class in self._y_classes:
                log_likelihood = np.sum(np.log(self._hypothesis(x, y_class)))
                y_log_prior_probability = np.log(self._y_prior_probabilities[y_class])
                y_log_posterior_probability = y_log_prior_probability + log_likelihood
                y_log_posterior_probabilities.append(y_log_posterior_probability)

            y_predicted = np.argmax(y_log_posterior_probabilities)
            Y_predicted.append(y_predicted)

        return np.array(Y_predicted)

    def _hypothesis(self,
                    x,
                    y_class):

        mean = self._y_means[y_class, :]
        variance = self._y_variances[y_class, :]

        return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance))
