import numpy as np


class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        predictions = []
        for row in X:
            distances = np.linalg.norm(self.X_train_ - row, axis=1)
            neighbor_idx = np.argsort(distances)[: self.n_neighbors]
            neighbor_labels = self.y_train_[neighbor_idx]
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class LogisticRegressionClassifier:
    def __init__(self, learning_rate=0.1, epochs=4000, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_encoded = np.array([np.where(self.classes_ == label)[0][0] for label in y])
        y_one_hot = np.eye(len(self.classes_))[y_encoded]

        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, size=(X.shape[1], len(self.classes_)))
        self.bias_ = np.zeros(len(self.classes_))

        for _ in range(self.epochs):
            logits = X @ self.weights_ + self.bias_
            probabilities = self._softmax(logits)
            error = probabilities - y_one_hot
            self.weights_ -= self.learning_rate * (X.T @ error) / X.shape[0]
            self.bias_ -= self.learning_rate * error.mean(axis=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        logits = X @ self.weights_ + self.bias_
        return self.classes_[np.argmax(logits, axis=1)]

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / exp_values.sum(axis=1, keepdims=True)


class DecisionTreeClassifier:
    def __init__(self, max_depth=4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        self.tree_ = self._build_tree(self.X_, self.y_, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray([self._predict_row(row, self.tree_) for row in X])

    def _build_tree(self, X, y, depth):
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            return {"label": self._majority_label(y)}

        feature_idx, threshold, gain = self._best_split(X, y)
        if gain <= 0:
            return {"label": self._majority_label(y)}

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _best_split(self, X, y):
        best_feature = 0
        best_threshold = 0.0
        best_gain = -1.0
        parent_gini = self._gini(y)

        for feature_idx in range(X.shape[1]):
            values = np.unique(X[:, feature_idx])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_weight = left_mask.mean()
                right_weight = right_mask.mean()
                child_gini = (
                    left_weight * self._gini(y[left_mask])
                    + right_weight * self._gini(y[right_mask])
                )
                gain = parent_gini - child_gini
                if gain > best_gain:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_gain = gain
        return best_feature, best_threshold, best_gain

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1.0 - np.sum(probabilities**2)

    def _majority_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict_row(self, row, node):
        if "label" in node:
            return node["label"]
        if row[node["feature_idx"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])


class ScaledClassifier:
    def __init__(self, classifier):
        self.scaler = StandardScaler()
        self.classifier = classifier

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.classifier.fit(X_scaled, y)
        return self

    def predict(self, X):
        return self.classifier.predict(self.scaler.transform(X))
