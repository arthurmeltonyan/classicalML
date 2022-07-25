import numpy as np


def split_train_validation_datasets(X,
                                    Y,
                                    validation_fraction=0.2):

    validation_indices = np.random.choice(len(X),
                                          size=round(len(X) * validation_fraction),
                                          replace=False)

    X_validation = X[validation_indices]
    Y_validation = Y[validation_indices]

    train_indices = list(set(range(0, len(X))) - set(validation_indices))

    X_train = X[train_indices]
    Y_train = Y[train_indices]

    return (X_train,
            X_validation,
            Y_train,
            Y_validation)
