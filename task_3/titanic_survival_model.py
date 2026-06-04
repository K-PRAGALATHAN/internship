from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path("titanic.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_titanic_model.joblib"
TARGET = "Survived"
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "CabinPresent"]


def extract_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    return title_map.get(title, title)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["Title"] = prepared["Name"].apply(extract_title)
    prepared["FamilySize"] = prepared["SibSp"] + prepared["Parch"] + 1
    prepared["IsAlone"] = (prepared["FamilySize"] == 1).astype(int)
    prepared["CabinPresent"] = prepared["Cabin"].notna().map({True: "Yes", False: "No"})
    return prepared


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def save_eda_plots(df: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Survival Class Distribution")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "survival_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Sex", hue=TARGET)
    plt.title("Survival by Sex")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "survival_by_sex.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Pclass", hue=TARGET)
    plt.title("Survival by Passenger Class")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "survival_by_pclass.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="Age", hue=TARGET, bins=30, kde=True)
    plt.title("Age Distribution by Survival")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "age_by_survival.png", dpi=160)
    plt.close()


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def save_feature_importance(best_model: Pipeline) -> pd.DataFrame:
    feature_names = get_feature_names(best_model)
    estimator = best_model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = abs(estimator.coef_[0])

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance_df, x="importance", y="feature")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=160)
    plt.close()

    return importance_df


def train_and_evaluate(df: pd.DataFrame):
    prepared = prepare_features(df)
    X = prepared.drop(columns=DROP_COLUMNS)
    y = prepared[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    fitted_models = {}
    predictions_by_model = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )
        fitted_models[name] = model
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values(
        by=["f1", "accuracy", "precision", "recall"],
        ascending=False,
    )
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions_by_model[best_name]

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODEL_PATH)

    labels = [0, 1]
    cm = confusion_matrix(y_test, best_predictions, labels=labels)
    display = ConfusionMatrixDisplay(cm, display_labels=["Not Survived", "Survived"])
    fig, ax = plt.subplots(figsize=(7, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    report = classification_report(y_test, best_predictions, zero_division=0)
    importance_df = save_feature_importance(best_model)
    return results_df, best_name, report, importance_df


def example_prediction() -> int:
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 28.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 7.8958,
                "Embarked": "S",
                "Title": "Mr",
                "FamilySize": 1,
                "IsAlone": 1,
                "CabinPresent": "No",
            }
        ]
    )
    return int(model.predict(sample)[0])


def main() -> None:
    df = load_data()
    prepared = prepare_features(df)

    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nSurvival counts:")
    print(df[TARGET].value_counts())
    print("\nEngineered title counts:")
    print(prepared["Title"].value_counts())

    save_eda_plots(prepared)
    results_df, best_name, report, importance_df = train_and_evaluate(df)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("\nClassification report:")
    print(report)
    print("\nTop feature importance:")
    print(importance_df.to_string(index=False))
    print(f"\nSaved model: {MODEL_PATH}")
    print("Example prediction:", example_prediction())


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


if __name__ == "__main__":
    main()
