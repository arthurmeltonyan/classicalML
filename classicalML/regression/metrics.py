import numpy as np


def root_mean_squared_error(Y_predicted,
                            Y_actual):

    return np.sqrt(np.mean((Y_predicted - Y_actual) ** 2))


def root_median_squared_error(Y_predicted,
                              Y_actual):

    return np.sqrt(np.median((Y_predicted - Y_actual) ** 2))


def mean_absolute_error(Y_predicted,
                        Y_actual):

    return np.mean(np.abs(Y_predicted - Y_actual))


def median_absolute_error(Y_predicted,
                          Y_actual):

    return np.median(np.abs(Y_predicted - Y_actual))


def maximum_absolute_error(Y_predicted,
                           Y_actual):

    return np.max(np.abs(Y_predicted - Y_actual))
