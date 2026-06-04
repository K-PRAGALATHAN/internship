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


DATA_PATH = Path("IRIS.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "iris_best_model.joblib"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


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
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
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


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def save_eda_plots(df: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    pairplot = sns.pairplot(df, hue=TARGET, diag_kind="hist", corner=True)
    pairplot.fig.suptitle("Iris Feature Separability", y=1.02)
    pairplot.savefig(ARTIFACTS_DIR / "iris_pairplot.png", dpi=160, bbox_inches="tight")
    plt.close(pairplot.fig)

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
    plt.close()


def build_models():
    return {
        "k-NN": ScaledClassifier(KNNClassifier(n_neighbors=5)),
        "Logistic Regression": ScaledClassifier(
            LogisticRegressionClassifier(learning_rate=0.1, epochs=4000, random_state=42)
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=4),
    }


def train_and_evaluate(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split_stratified(df)

    results = []
    fitted_models = {}
    predictions_by_model = {}
    for name, model in build_models().items():
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
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions_by_model[best_name]

    per_class_report = precision_recall_by_class(y_test.to_numpy(), best_predictions)
    cm = confusion_matrix_df(y_test.to_numpy(), best_predictions)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODEL_PATH)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    return results_df, best_name, per_class_report, cm


def predict_example() -> str:
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURES,
    )
    return model.predict(sample.to_numpy())[0]


def main() -> None:
    df = load_data()
    print("Dataset shape:", df.shape)
    print("\nClass counts:")
    print(df[TARGET].value_counts())
    print("\nMissing values:")
    print(df.isna().sum())

    save_eda_plots(df)
    results_df, best_name, per_class_report, cm = train_and_evaluate(df)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("\nPrecision/recall by class:")
    print(per_class_report.to_string(index=False))
    print("\nConfusion matrix:")
    print(cm)
    print(f"\nSaved model: {MODEL_PATH}")
    print("Example prediction:", predict_example())


if __name__ == "__main__":
    main()
