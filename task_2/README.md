# House Price Prediction

This project builds a regression model to predict house prices using housing features such as living area, lot size, bedrooms, bathrooms, location, condition, and year built.

## Goal

Build a regression model to predict house prices with focus on:

- Feature engineering
- Handling missing values
- Feature transformation
- Model selection
- Model evaluation
- Residual analysis

## Dataset

Dataset file:

```text
data.csv
```

Target column:

```text
price
```

The dataset contains:

- 4,600 rows
- 18 columns
- No missing values

During preprocessing, 49 rows with `price <= 0` were removed because they are invalid for price prediction and log transformation.

## Features Used

Original useful features include:

- `bedrooms`
- `bathrooms`
- `sqft_living`
- `sqft_lot`
- `floors`
- `waterfront`
- `view`
- `condition`
- `sqft_above`
- `sqft_basement`
- `yr_built`
- `yr_renovated`
- `city`
- `statezip`
- `date`

Engineered features:

- `sale_year`
- `sale_month`
- `zipcode`
- `house_age`
- `was_renovated`

## Preprocessing

The preprocessing pipeline includes:

- Removing rows with invalid target values
- Parsing the `date` column
- Extracting zipcode from `statezip`
- Creating house age and renovation indicator features
- Dropping unnecessary/high-cardinality columns:
  - `street`
  - `country`
  - `date`
  - `statezip`
- Median imputation for numerical features
- Mode imputation for categorical features
- Log transformation for skewed numerical features
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Log transformation of the target variable using `log1p(price)`

## Models Compared

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics

Models were evaluated using:

- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error
- Residual analysis plot

## Results

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Linear Regression | 173631.33 | 87215.51 |
| Gradient Boosting | 220853.56 | 110868.85 |
| Random Forest | 225956.71 | 112903.59 |

The best model is **Linear Regression**, selected based on the lowest RMSE.

## Project Structure

```text
task_2/
  data.csv
  README.md
  house_price_prediction.ipynb
  house_price_model.py
  create_house_notebook.py
  artifacts/
    best_house_price_model.joblib
    model_comparison.csv
    price_distribution.png
    log_price_distribution.png
    price_vs_sqft_living.png
    correlation_heatmap.png
    residual_plot.png
```

## Visualizations

The notebook and script generate:

- Price distribution plot
- Log-transformed price distribution plot
- Price vs living area scatter plot
- Correlation heatmap
- Residual analysis plot

All generated plots are saved in:

```text
artifacts/
```

## How To Run

Install the required packages:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib nbformat nbclient ipykernel
```

Run the complete training and evaluation pipeline:

```bash
python house_price_model.py
```

Or open and run the notebook:

```text
house_price_prediction.ipynb
```

## Model Inference

The best model is saved at:

```text
artifacts/best_house_price_model.joblib
```

Example prediction code:

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/best_house_price_model.joblib")

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

predicted_price = model.predict(sample)[0]
print(f"Predicted price: ${predicted_price:,.2f}")
```

Example output:

```text
Predicted price: $616,953.33
```

## Deliverables

- Notebook with preprocessing, models, evaluation, plots, and prediction example
- Saved model file
- README with instructions to run prediction
