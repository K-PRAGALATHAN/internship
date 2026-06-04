from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = "house_price_prediction.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


cells = [
    markdown(
        """
        # House Price Prediction

        Goal: build a regression model to predict house prices with feature engineering,
        missing-value handling, transformations, model selection, and residual analysis.
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
        from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, root_mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

        sns.set_theme(style="whitegrid")

        DATA_PATH = Path("data.csv")
        ARTIFACTS_DIR = Path("artifacts")
        MODEL_PATH = ARTIFACTS_DIR / "best_house_price_model.joblib"
        TARGET = "price"
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
    code("df.info()"),
    code(
        """
        print("Missing values:")
        print(df.isna().sum())

        print("\\nRows with price <= 0:", (df[TARGET] <= 0).sum())
        """
    ),
    code("df[df[TARGET] > 0].describe().T"),
    markdown("## EDA"),
    code(
        """
        valid_df = df[df[TARGET] > 0].copy()

        plt.figure(figsize=(8, 5))
        sns.histplot(valid_df[TARGET], bins=50, kde=True)
        plt.title("House Price Distribution")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "price_distribution.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(np.log1p(valid_df[TARGET]), bins=50, kde=True)
        plt.title("Log-Transformed Price Distribution")
        plt.xlabel("log1p(price)")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "log_price_distribution.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=valid_df, x="sqft_living", y=TARGET, hue="city", legend=False, alpha=0.65)
        plt.title("Price vs Living Area")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "price_vs_sqft_living.png", dpi=160)
        plt.show()
        """
    ),
    code(
        """
        numeric_corr = valid_df.select_dtypes(include=[np.number]).corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_corr, cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "correlation_heatmap.png", dpi=160)
        plt.show()
        """
    ),
    markdown("## Feature Engineering"),
    code(
        """
        def prepare_features(data):
            prepared = data.copy()
            prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
            prepared["sale_year"] = prepared["date"].dt.year
            prepared["sale_month"] = prepared["date"].dt.month
            prepared["zipcode"] = prepared["statezip"].str.extract(r"(\\d{5})", expand=False)
            prepared["house_age"] = prepared["sale_year"] - prepared["yr_built"]
            prepared["was_renovated"] = (prepared["yr_renovated"] > 0).astype(int)
            return prepared


        prepared = prepare_features(df)
        prepared = prepared[prepared[TARGET] > 0].copy()
        prepared.head()
        """
    ),
    markdown("## Preprocessing"),
    code(
        """
        drop_columns = ["price", "street", "country", "date", "statezip"]
        log_numeric_features = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]
        other_numeric_features = [
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
        categorical_features = ["city", "zipcode"]

        X = prepared.drop(columns=drop_columns)
        y = prepared[TARGET]

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

        preprocessor = ColumnTransformer(
            transformers=[
                ("log_numeric", log_numeric_pipeline, log_numeric_features),
                ("numeric", numeric_pipeline, other_numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
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
        )

        X_train.shape, X_test.shape
        """
    ),
    markdown("## Model Training And Comparison"),
    code(
        """
        def make_model(regressor):
            return TransformedTargetRegressor(
                regressor=Pipeline(
                    steps=[
                        ("preprocess", preprocessor),
                        ("model", regressor),
                    ]
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            )


        models = {
            "Linear Regression": make_model(LinearRegression()),
            "Random Forest": make_model(
                RandomForestRegressor(
                    n_estimators=120,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=18,
                )
            ),
            "Gradient Boosting": make_model(
                GradientBoostingRegressor(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                )
            ),
        }

        results = []
        fitted_models = {}
        predictions_by_model = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = np.maximum(model.predict(X_test), 0)
            results.append(
                {
                    "model": name,
                    "rmse": root_mean_squared_error(y_test, predictions),
                    "mae": mean_absolute_error(y_test, predictions),
                }
            )
            fitted_models[name] = model
            predictions_by_model[name] = predictions

        results_df = pd.DataFrame(results).sort_values(["rmse", "mae"])
        results_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
        results_df
        """
    ),
    markdown("## Residual Analysis"),
    code(
        """
        best_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_name]
        best_predictions = predictions_by_model[best_name]
        residuals = y_test - best_predictions

        print("Best model:", best_name)
        print("RMSE:", root_mean_squared_error(y_test, best_predictions))
        print("MAE:", mean_absolute_error(y_test, best_predictions))
        """
    ),
    code(
        """
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=best_predictions, y=residuals, alpha=0.65)
        plt.axhline(0, color="red", linestyle="--", linewidth=1)
        plt.title(f"Residual Analysis - {best_name}")
        plt.xlabel("Predicted Price")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "residual_plot.png", dpi=160)
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
    markdown("## Example Prediction"),
    code(
        """
        loaded_model = joblib.load(MODEL_PATH)

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

        predicted_price = loaded_model.predict(sample)[0]
        print(f"Predicted price: ${predicted_price:,.2f}")
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
