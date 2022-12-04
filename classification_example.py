from sklearn import datasets

from classicalML.dataset import cross_validation as cv
from classicalML.classification import metrics
from classicalML.classification.models import linear_model as lm
from classicalML.classification.models import decision_tree as dt
from classicalML.classification.models import nearest_neighbors as knn
from classicalML.classification.models import random_forest as rf
from classicalML.classification.models import support_vector_machine as svm
from classicalML.classification.models import naive_bayes as nb


def check_classifier(X,
                     Y,
                     classifier):

    X_train, X_validation, Y_train, Y_validation = cv.split_train_validation_datasets(X,
                                                                                      Y,
                                                                                      validation_fraction=0.2)
    classifier.fit(X_train,
                   Y_train)
    Y_predicted = classifier.predict(X_validation)
    print(classifier)
    print(f'confusion matrix: \n {metrics.confusion_matrix(Y_predicted, Y_validation)}')
    for metric in [metrics.accuracy_score,
                   metrics.error_score,
                   metrics.precision_score,
                   metrics.correctness_score,
                   metrics.markedness_score,
                   metrics.recall_score,
                   metrics.specificity_score,
                   metrics.fallout_score,
                   metrics.missrate_score,
                   metrics.jaccrad_score,
                   metrics.informedness_score,
                   metrics.balanced_accuracy_score,
                   metrics.positive_likelihood_ratio,
                   metrics.negative_likelihood_ratio,
                   metrics.matthews_correlation_score]:
        metric_name = str(metric).split(' ')[1]
        metric_value = metric(Y_predicted, Y_validation)
        print(f'{metric_name} = {metric_value}')
    print(f'f_0.5 = {metrics.f_score(Y_predicted, Y_validation, beta=0.5)}')
    print(f'f_1.0 = {metrics.f_score(Y_predicted, Y_validation, beta=1.0)}')
    print(f'f_2.0 = {metrics.f_score(Y_predicted, Y_validation, beta=2.0)}')
    print()


def main():
    classification_dataset = datasets.make_classification(n_samples=10000,
                                                          n_features=10,
                                                          random_state=2020)
    check_classifier(*classification_dataset,
                     lm.LinearClassifier(alpha=0.001, epochs=1000))
    check_classifier(*classification_dataset,
                     dt.DecisionTreeClassifier(max_depth=8, min_samples=20))
    check_classifier(*classification_dataset,
                     knn.KNearestNeighborsClassifier(k=5))
    check_classifier(*classification_dataset,
                     rf.RandomForestClassifier(estimators=2, max_depth=8, min_samples=20))
    check_classifier(*classification_dataset,
                     svm.LinearSupportVectorClassifier(alpha=0.001, epochs=1000, parameter=1.0))
    check_classifier(*classification_dataset,
                     nb.NaiveBayesClassifier())


if __name__ == '__main__':
    main()
