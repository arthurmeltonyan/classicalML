import numpy as np


class LeafNode:

    def __init__(self,
                 value):

        self.value = value


class BranchNode:

    def __init__(self,
                 feature,
                 threshold,
                 left,
                 right):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeClassifier:

    def __init__(self,
                 max_depth=5,
                 min_samples=5):

        self._tree = None
        self._max_depth = max_depth
        self._min_samples = min_samples

    def __str__(self):

        return f'DecisionTreeClassifier' \
               f'(max_depth={self._max_depth}, ' \
               f' min_samples={self._min_samples})'

    @staticmethod
    def _entropy(Y_subset):

        values, counts = np.unique(Y_subset,
                                   return_counts=True)
        total_count = np.sum(counts)
        probabilities = counts / total_count

        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def _impurity(Y_left_subset,
                  Y_right_subset):

        Y_subset = np.concatenate([Y_left_subset, Y_right_subset])
        entropy = DecisionTreeClassifier._entropy(Y_subset)
        left_probability = Y_left_subset.shape[0] / Y_subset.shape[0]
        left_entropy = DecisionTreeClassifier._entropy(Y_left_subset)
        left_term = left_probability * left_entropy
        right_probability = Y_right_subset.shape[0] / Y_subset.shape[0]
        right_entropy = DecisionTreeClassifier._entropy(Y_right_subset)
        right_term = right_probability * right_entropy

        return entropy - (left_term + right_term)

    @staticmethod
    def _predict_for_sample(Y_subset):

        return np.bincount(Y_subset).argmax()

    def _create_node(self,
                     X_subset,
                     Y_subset,
                     depth):

        if depth == self._max_depth or Y_subset.shape[0] < self._min_samples:
            y_predicted = self._predict_for_sample(Y_subset)
            return LeafNode(value=y_predicted)

        max_feature = 0
        values = X_subset.transpose()[max_feature]
        unique_values = np.sort(np.unique(values))
        max_threshold = (unique_values[0] + unique_values[1]) / 2
        max_left_indices = np.argwhere(values <= max_threshold).flatten()
        max_right_indices = np.argwhere(values > max_threshold).flatten()
        max_information_gain = self._impurity(Y_subset[max_left_indices],
                                              Y_subset[max_right_indices])

        for feature, values in enumerate(X_subset.transpose()):

            unique_values = np.sort(np.unique(values))
            thresholds = (unique_values[1:] + unique_values[:-1]) / 2
            for threshold in thresholds:
                left_indices = np.argwhere(values <= threshold).flatten()
                right_indices = np.argwhere(values > threshold).flatten()
                information_gain = self._impurity(Y_subset[left_indices],
                                                  Y_subset[right_indices])
                if max_information_gain < information_gain:
                    max_feature = feature
                    max_threshold = threshold
                    max_left_indices = left_indices
                    max_right_indices = right_indices
                    max_information_gain = information_gain

        left_node = self._create_node(X_subset[max_left_indices],
                                      Y_subset[max_left_indices],
                                      depth + 1)
        right_node = self._create_node(X_subset[max_right_indices],
                                       Y_subset[max_right_indices],
                                       depth + 1)

        return BranchNode(feature=max_feature,
                          threshold=max_threshold,
                          left=left_node,
                          right=right_node)

    def fit(self,
            X_train,
            Y_train):

        self._tree = self._create_node(X_train,
                                       Y_train,
                                       1)

    def predict(self,
                X):

        Y_predicted = []

        for x in X:

            node = self._tree
            while isinstance(node, BranchNode):
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            y_predicted = node.value
            Y_predicted.append(y_predicted)

        return np.array(Y_predicted)
