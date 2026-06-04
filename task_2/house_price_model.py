from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


DATA_PATH = Path("data.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_house_price_model.joblib"
TARGET = "price"
DROP_COLUMNS = ["price", "street", "country", "date", "statezip"]
LOG_NUMERIC_FEATURES = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]
OTHER_NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "floors",
    "waterfront",
    "view",
    "condition",
    "yr_built",
    "yr_renovated",
    "sale_year",
    "sale_month",
    "house_age",
    "was_renovated",
]
CATEGORICAL_FEATURES = ["city", "zipcode"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["sale_year"] = prepared["date"].dt.year
    prepared["sale_month"] = prepared["date"].dt.month
    prepared["zipcode"] = prepared["statezip"].str.extract(r"(\d{5})", expand=False)
    prepared["house_age"] = prepared["sale_year"] - prepared["yr_built"]
    prepared["was_renovated"] = (prepared["yr_renovated"] > 0).astype(int)
    return prepared


def build_preprocessor() -> ColumnTransformer:
    log_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log_numeric", log_numeric_pipeline, LOG_NUMERIC_FEATURES),
            ("numeric", numeric_pipeline, OTHER_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, TransformedTargetRegressor]:
    return {
        "Linear Regression": TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    ("model", LinearRegression()),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=120,
                            random_state=42,
                            n_jobs=-1,
                            max_depth=18,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Gradient Boosting": TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=250,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def save_eda_plots(df: pd.DataFrame) -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=50, kde=True)
    plt.title("House Price Distribution")
    plt.xlabel("Price")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "price_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(np.log1p(df[TARGET]), bins=50, kde=True)
    plt.title("Log-Transformed Price Distribution")
    plt.xlabel("log1p(price)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "log_price_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="sqft_living", y=TARGET, hue="city", legend=False, alpha=0.65)
    plt.title("Price vs Living Area")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "price_vs_sqft_living.png", dpi=160)
    plt.close()

    numeric_corr = df.select_dtypes(include=[np.number]).corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_corr, cmap="coolwarm", center=0)
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def train_and_evaluate(df: pd.DataFrame):
    prepared = prepare_features(df)
    prepared = prepared[prepared[TARGET] > 0].copy()

    X = prepared.drop(columns=DROP_COLUMNS)
    y = prepared[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    fitted_models = {}
    predictions_by_model = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = np.maximum(predictions, 0)
        results.append(
            {
                "model": name,
                "rmse": root_mean_squared_error(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
            }
        )
        fitted_models[name] = model
        predictions_by_model[name] = predictions

    results_df = pd.DataFrame(results).sort_values(["rmse", "mae"], ascending=True)
    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = predictions_by_model[best_name]
    residuals = y_test - best_predictions

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
    joblib.dump(best_model, MODEL_PATH)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=best_predictions, y=residuals, alpha=0.65)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Residual Analysis - {best_name}")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "residual_plot.png", dpi=160)
    plt.close()

    return results_df, best_name, X_test, y_test, best_predictions


def example_prediction() -> float:
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "bedrooms": 3.0,
                "bathrooms": 2.0,
                "sqft_living": 1800,
                "sqft_lot": 5000,
                "floors": 1.0,
                "waterfront": 0,
                "view": 0,
                "condition": 3,
                "sqft_above": 1500,
                "sqft_basement": 300,
                "yr_built": 1995,
                "yr_renovated": 0,
                "city": "Seattle",
                "sale_year": 2014,
                "sale_month": 6,
                "zipcode": "98103",
                "house_age": 19,
                "was_renovated": 0,
            }
        ]
    )
    return float(model.predict(sample)[0])


def main() -> None:
    df = load_data()
    valid_df = df[df[TARGET] > 0].copy()

    print("Dataset shape:", df.shape)
    print("Rows with non-positive price removed for modeling:", len(df) - len(valid_df))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(valid_df.describe().T)

    save_eda_plots(valid_df)
    results_df, best_name, _, _, _ = train_and_evaluate(df)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Example predicted price: {example_prediction():,.2f}")


if __name__ == "__main__":
    main()
