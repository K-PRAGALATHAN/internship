# Titanic Survival Prediction

This project predicts Titanic passenger survival using passenger attributes such as class, sex, age, fare, family information, embarkation point, title, and cabin availability.

## Goal

Predict survival on the Titanic using passenger attributes.

The project emphasizes:

- Feature creation
- Missing-value handling
- Categorical encoding
- Classification model comparison
- Model explainability using feature importance
- Interface example for prediction

## Dataset

Dataset file:

```text
titanic.csv
```

Target column:

```text
Survived
```

Dataset summary:

- 418 rows
- 12 columns
- Target classes:
  - `0`: Not survived
  - `1`: Survived

Missing values:

| Column | Missing Values |
| --- | ---: |
| Cabin | 327 |
| Age | 86 |
| Fare | 1 |

## Feature Engineering

The following features were created:

- `Title`: extracted from passenger name, for example `Mr`, `Mrs`, `Miss`, `Master`
- `FamilySize`: calculated as `SibSp + Parch + 1`
- `IsAlone`: identifies whether the passenger travelled alone
- `CabinPresent`: identifies whether cabin information is available

Rare titles were grouped into a common `Rare` category.

## Missing Data Strategy

Missing values were handled inside preprocessing pipelines:

- `Age`: median imputation
- `Fare`: median imputation
- `Cabin`: converted into `CabinPresent`
- Categorical columns: most frequent value imputation

## Preprocessing

The preprocessing pipeline includes:

- Numerical feature scaling using `StandardScaler`
- Categorical encoding using `OneHotEncoder`
- Median imputation for numerical columns
- Mode imputation for categorical columns
- Dropping columns not used directly by the model:
  - `PassengerId`
  - `Name`
  - `Ticket`
  - `Cabin`

## Models Compared

The following classification models were trained and compared:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Results

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The selected best model is **Logistic Regression**.

## Model Explainability

Feature importance was used for explanation.

Top influential features from the selected model include:

- `Sex`
- `Title`
- `IsAlone`
- `Embarked`
- `Fare`
- `SibSp`
- `Pclass`
- `CabinPresent`

The feature importance plot is saved as:

```text
artifacts/feature_importance.png
```

## Project Structure

```text
task_3/
  titanic.csv
  README.md
  titanic_survival_prediction.ipynb
  titanic_survival_model.py
  create_titanic_notebook.py
  artifacts/
    best_titanic_model.joblib
    model_comparison.csv
    feature_importance.csv
    feature_importance.png
    confusion_matrix.png
    survival_distribution.png
    survival_by_sex.png
    survival_by_pclass.png
    age_by_survival.png
```

## How To Run

Install the required packages:

```bash
python -m pip install pandas matplotlib seaborn scikit-learn joblib nbformat nbclient ipykernel
```

Run the complete training pipeline:

```bash
python titanic_survival_model.py
```

Or open and run the notebook:

```text
titanic_survival_prediction.ipynb
```

## Interface Example

The best model is saved at:

```text
artifacts/best_titanic_model.joblib
```

Example prediction code:

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/best_titanic_model.joblib")

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

prediction = model.predict(sample)[0]
label = "Survived" if prediction == 1 else "Not survived"
print(label)
```

Example output:

```text
Not survived
```

## Deliverables

- Notebook with preprocessing, models, evaluation, and explanation
- Saved model file
- README with interface example
