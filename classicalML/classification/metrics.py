import collections

import numpy as np


def confusion_matrix(Y_actual,
                     Y_predicted):

    matrix = collections.defaultdict(int)

    for y_actual, y_predicted in zip(Y_actual,
                                     Y_predicted):
        if y_actual == 0 and y_predicted == 0:
            matrix['true_negative'] += 1
        elif y_actual == 1 and y_predicted == 1:
            matrix['true_positive'] += 1
        elif y_actual == 0 and y_predicted == 1:
            matrix['false_positive'] += 1
        elif y_actual == 1 and y_predicted == 0:
            matrix['false_negative'] += 1

    matrix = dict(matrix)

    return matrix


def accuracy_score(Y_actual,
                   Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    actual_true = matrix['true_positive'] + matrix['true_negative']
    actual_false = matrix['false_positive'] + matrix['false_negative']
    true_ratio = actual_true / (actual_true + actual_false)

    return true_ratio


def error_score(Y_actual,
                Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    actual_false = matrix['false_positive'] + matrix['false_negative']
    actual_true = matrix['true_positive'] + matrix['true_negative']
    false_ratio = actual_false / (actual_false + actual_true)

    return false_ratio


def precision_score(Y_actual,
                    Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    true_positive = matrix['true_positive']
    predicted_positive = matrix['true_positive'] + matrix['false_positive']
    positive_predictive_value = true_positive / predicted_positive

    return positive_predictive_value


def correctness_score(Y_actual,
                      Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    true_negative = matrix['true_negative']
    predicted_negative = matrix['true_negative'] + matrix['false_negative']
    negative_predictive_value = true_negative / predicted_negative

    return negative_predictive_value


def markedness_score(Y_actual,
                     Y_predicted):

    positive_predictive_value = precision_score(Y_actual,
                                                Y_predicted)
    negative_predictive_value = correctness_score(Y_actual,
                                                  Y_predicted)

    return positive_predictive_value + negative_predictive_value - 1


def recall_score(Y_actual,
                 Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    true_positive = matrix['true_positive']
    actual_positive = matrix['true_positive'] + matrix['false_negative']
    true_positive_rate = true_positive / actual_positive

    return true_positive_rate


def specificity_score(Y_actual,
                      Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    true_negative = matrix['true_negative']
    actual_negative = matrix['true_negative'] + matrix['false_positive']
    true_negative_rate = true_negative / actual_negative

    return true_negative_rate


def fallout_score(Y_actual,
                  Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    false_positive = matrix['false_positive']
    actual_negative = matrix['false_positive'] + matrix['true_negative']
    false_positive_rate = false_positive / actual_negative

    return false_positive_rate


def missrate_score(Y_actual,
                   Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    false_negative = matrix['false_negative']
    actual_positive = matrix['false_negative'] + matrix['true_positive']
    false_negative_rate = false_negative / actual_positive

    return false_negative_rate


def jaccrad_score(Y_actual,
                  Y_predicted):

    matrix = confusion_matrix(Y_actual,
                              Y_predicted)
    true_positive = matrix['true_positive']
    not_false_negative = matrix['false_negative'] + matrix['true_positive'] + matrix['false_positive']

    return true_positive / not_false_negative


def informedness_score(Y_actual,
                       Y_predicted):

    true_positive_rate = recall_score(Y_actual,
                                      Y_predicted)
    true_negative_rate = specificity_score(Y_actual,
                                           Y_predicted)

    return true_positive_rate + true_negative_rate - 1


def balanced_accuracy_score(Y_actual,
                            Y_predicted):

    true_positive_rate = recall_score(Y_actual,
                                      Y_predicted)
    true_negative_rate = specificity_score(Y_actual,
                                           Y_predicted)

    return (true_positive_rate + true_negative_rate) / 2


def positive_likelihood_ratio(Y_actual,
                              Y_predicted):

    true_positive_rate = recall_score(Y_actual,
                                      Y_predicted)
    false_positive_rate = fallout_score(Y_actual,
                                        Y_predicted)

    return true_positive_rate / false_positive_rate


def negative_likelihood_ratio(Y_actual,
                              Y_predicted):

    false_negative_rate = missrate_score(Y_actual,
                                         Y_predicted)
    true_negative_rate = specificity_score(Y_actual,
                                           Y_predicted)

    return false_negative_rate / true_negative_rate


def matthews_correlation_score(Y_actual,
                               Y_predicted):

    correlation = np.correlate(Y_actual,
                               Y_predicted)[0]

    return correlation


def f_score(Y_actual,
            Y_predicted,
            beta=1):

    true_positive_rate = recall_score(Y_actual,
                                      Y_predicted)
    positive_predictive_value = precision_score(Y_actual,
                                                Y_predicted)

    return (1 + beta ** 2) / (1 / true_positive_rate + (beta ** 2) / positive_predictive_value)