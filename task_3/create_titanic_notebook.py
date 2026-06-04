from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = "titanic_survival_prediction.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # Titanic Survival Prediction

        Goal: predict survival on the Titanic using passenger attributes.
        The notebook emphasizes title extraction, family-size features, cabin presence,
        missing-value handling, categorical encoding, model evaluation, and feature-importance explanation.
        """
    ),
    code(
        """
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

        sns.set_theme(style="whitegrid")

        DATA_PATH = Path("titanic.csv")
        ARTIFACTS_DIR = Path("artifacts")
        MODEL_PATH = ARTIFACTS_DIR / "best_titanic_model.joblib"
        TARGET = "Survived"
        ARTIFACTS_DIR.mkdir(exist_ok=True)
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
    code(
        """
        df.info()
        """
    ),
    code(
        """
        print("Missing values:")
        print(df.isna().sum().sort_values(ascending=False))

        print("\\nSurvival counts:")
        print(df[TARGET].value_counts())
        """
    ),
    markdown("## Feature Engineering"),
    code(
        """
        def extract_title(name):
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


        def prepare_features(data):
            prepared = data.copy()
            prepared["Title"] = prepared["Name"].apply(extract_title)
            prepared["FamilySize"] = prepared["SibSp"] + prepared["Parch"] + 1
            prepared["IsAlone"] = (prepared["FamilySize"] == 1).astype(int)
            prepared["CabinPresent"] = prepared["Cabin"].notna().map({True: "Yes", False: "No"})
            return prepared


        prepared = prepare_features(df)
        prepared[["Name", "Title", "SibSp", "Parch", "FamilySize", "IsAlone", "Cabin", "CabinPresent"]].head()
        """
    ),
    code(
        """
        prepared["Title"].value_counts()
        """
    ),
    markdown("## EDA"),
    code(
        """
        plt.figure(figsize=(7, 5))
        sns.countplot(data=prepared, x=TARGET)
        plt.title("Survival Class Distribution")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "survival_distribution.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 5))
        sns.countplot(data=prepared, x="Sex", hue=TARGET)
        plt.title("Survival by Sex")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "survival_by_sex.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(7, 5))
        sns.countplot(data=prepared, x="Pclass", hue=TARGET)
        plt.title("Survival by Passenger Class")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "survival_by_pclass.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(data=prepared, x="Age", hue=TARGET, bins=30, kde=True)
        plt.title("Age Distribution by Survival")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "age_by_survival.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Preprocessing"),
    code(
        """
        drop_columns = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
        numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
        categorical_features = ["Sex", "Embarked", "Title", "CabinPresent"]

        X = prepared.drop(columns=drop_columns)
        y = prepared[TARGET]

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

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )
        """
    ),
    markdown("## Train/Test Split"),
    code(
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_train.shape, X_test.shape
        """
    ),
    markdown("## Train And Compare Models"),
    code(
        """
        models = {
            "Logistic Regression": Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            ),
            "Random Forest": Pipeline(
                steps=[
                    ("preprocess", preprocessor),
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
                    ("preprocess", preprocessor),
                    ("model", GradientBoostingClassifier(random_state=42)),
                ]
            ),
        }

        results = []
        fitted_models = {}
        predictions_by_model = {}

        for name, model in models.items():
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
        results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
        results_df
        """
    ),
    markdown("## Final Metrics"),
    code(
        """
        best_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_name]
        best_predictions = predictions_by_model[best_name]

        print("Best model:", best_name)
        print(classification_report(y_test, best_predictions, zero_division=0))
        """
    ),
    code(
        """
        cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(cm, display_labels=["Not Survived", "Survived"])
        fig, ax = plt.subplots(figsize=(7, 6))
        display.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {best_name}")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Model Explanation With Feature Importance"),
    code(
        """
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        estimator = best_model.named_steps["model"]

        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_
        else:
            importance_values = abs(estimator.coef_[0])

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importance_values})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        importance_df.to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
        importance_df
        """
    ),
    code(
        """
        plt.figure(figsize=(9, 6))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title("Top Feature Importance")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Save Model"),
    code(
        """
        joblib.dump(best_model, MODEL_PATH)
        MODEL_PATH
        """
    ),
    markdown("## Interface Example"),
    code(
        """
        loaded_model = joblib.load(MODEL_PATH)

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

        prediction = loaded_model.predict(sample)[0]
        label = "Survived" if prediction == 1 else "Not survived"
        print(label)
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
