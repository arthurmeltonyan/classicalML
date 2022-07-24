import random


def split_train_validation_datasets(X,
                                    Y,
                                    ratio=0.2):

    validation_indices = random.sample(list(range(0, len(X))),
                                       k=round(len(X) * ratio))

    X_validation = [X[index]
                    for index in validation_indices]
    Y_validation = [Y[index]
                    for index in validation_indices]

    train_indices = list(set(range(0, len(X))) - set(validation_indices))

    X_train = [X[index]
               for index in train_indices]
    Y_train = [Y[index]
               for index in train_indices]

    return X_train, X_validation, Y_train, Y_validation
