from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = "iris_classification.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # Iris Species Classification

        Goal: build a classification model to predict iris species from sepal and petal measurements.
        This notebook performs EDA, visualizes class separability, compares k-NN, Logistic Regression,
        and Decision Tree models, reports accuracy/confusion matrix/precision/recall, and saves the best model.
        """
    ),
    code(
        """
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from iris_models import (
            DecisionTreeClassifier,
            KNNClassifier,
            LogisticRegressionClassifier,
            ScaledClassifier,
        )

        sns.set_theme(style="whitegrid")

        DATA_PATH = Path("IRIS.csv")
        ARTIFACTS_DIR = Path("artifacts")
        MODEL_PATH = ARTIFACTS_DIR / "iris_best_model.joblib"
        FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        TARGET = "species"
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        """
    ),
    markdown("## Helper Functions"),
    code(
        """
        def train_test_split_stratified(df, test_size=0.2, random_state=42):
            rng = np.random.default_rng(random_state)
            train_parts = []
            test_parts = []
            for _, group in df.groupby(TARGET):
                indices = group.index.to_numpy().copy()
                rng.shuffle(indices)
                test_count = int(round(len(indices) * test_size))
                test_parts.append(df.loc[indices[:test_count]])
                train_parts.append(df.loc[indices[test_count:]])
            train_df = pd.concat(train_parts).sample(frac=1, random_state=random_state)
            test_df = pd.concat(test_parts).sample(frac=1, random_state=random_state)
            return train_df[FEATURES], test_df[FEATURES], train_df[TARGET], test_df[TARGET]


        def accuracy_score(y_true, y_pred):
            return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


        def precision_recall_by_class(y_true, y_pred):
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            rows = []
            for label in labels:
                true_positive = np.sum((y_true == label) & (y_pred == label))
                false_positive = np.sum((y_true != label) & (y_pred == label))
                false_negative = np.sum((y_true == label) & (y_pred != label))
                precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
                recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
                rows.append({"species": label, "precision": precision, "recall": recall})
            return pd.DataFrame(rows)


        def confusion_matrix_df(y_true, y_pred):
            labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
            matrix = pd.DataFrame(0, index=labels, columns=labels)
            for actual, predicted in zip(y_true, y_pred):
                matrix.loc[actual, predicted] += 1
            return matrix
        """
    ),
    markdown("## Load Data"),
    code(
        """
        df = pd.read_csv(DATA_PATH)
        print(df.shape)
        df.head()
        """
    ),
    code("df.info()"),
    code("df.describe()"),
    code(
        """
        print("Missing values:")
        print(df.isna().sum())
        print("\\nClass counts:")
        print(df[TARGET].value_counts())
        """
    ),
    markdown("## EDA And Class Separability"),
    code(
        """
        pairplot = sns.pairplot(df, hue=TARGET, diag_kind="hist", corner=True)
        pairplot.fig.suptitle("Iris Feature Separability", y=1.02)
        pairplot.savefig(ARTIFACTS_DIR / "iris_pairplot.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="petal_length",
            y="petal_width",
            hue=TARGET,
            style=TARGET,
            s=80,
        )
        plt.title("Petal Length vs Petal Width by Species")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "petal_scatter.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Train/Test Split"),
    code(
        """
        X_train, X_test, y_train, y_test = train_test_split_stratified(df)
        X_train.shape, X_test.shape
        """
    ),
    markdown("## Train And Compare Algorithms"),
    code(
        """
        models = {
            "k-NN": ScaledClassifier(KNNClassifier(n_neighbors=5)),
            "Logistic Regression": ScaledClassifier(
                LogisticRegressionClassifier(learning_rate=0.1, epochs=4000, random_state=42)
            ),
            "Decision Tree": DecisionTreeClassifier(max_depth=4),
        }

        results = []
        fitted_models = {}
        predictions_by_model = {}

        for name, model in models.items():
            model.fit(X_train.to_numpy(), y_train.to_numpy())
            predictions = model.predict(X_test.to_numpy())
            pr = precision_recall_by_class(y_test.to_numpy(), predictions)
            results.append(
                {
                    "model": name,
                    "accuracy": accuracy_score(y_test.to_numpy(), predictions),
                    "precision_macro": pr["precision"].mean(),
                    "recall_macro": pr["recall"].mean(),
                }
            )
            fitted_models[name] = model
            predictions_by_model[name] = predictions

        results_df = pd.DataFrame(results).sort_values(
            by=["accuracy", "precision_macro", "recall_macro", "model"],
            ascending=[False, False, False, True],
        )
        results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
        results_df
        """
    ),
    markdown("## Best Model Metrics"),
    code(
        """
        best_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_name]
        best_predictions = predictions_by_model[best_name]

        print("Best model:", best_name)
        print("Accuracy:", accuracy_score(y_test.to_numpy(), best_predictions))

        per_class_report = precision_recall_by_class(y_test.to_numpy(), best_predictions)
        per_class_report
        """
    ),
    code(
        """
        cm = confusion_matrix_df(y_test.to_numpy(), best_predictions)
        cm
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {best_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Save Best Model"),
    code(
        """
        joblib.dump(best_model, MODEL_PATH)
        MODEL_PATH
        """
    ),
    markdown("## Example Inference"),
    code(
        """
        loaded_model = joblib.load(MODEL_PATH)
        sample = pd.DataFrame(
            [[5.1, 3.5, 1.4, 0.2]],
            columns=FEATURES,
        )
        loaded_model.predict(sample.to_numpy())[0]
        """
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as file:
    nbf.write(nb, file)

print(f"Created {NOTEBOOK_PATH}")
