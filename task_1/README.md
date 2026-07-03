# Iris Species Classification

## Goal

Build a classification model to predict iris flower species using the classic Iris dataset features:

- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is `species`.

## Dataset

The dataset used is `IRIS.csv`.

It contains 150 records from three iris species:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

## Project Workflow

1. Loaded and explored the dataset.
2. Performed Exploratory Data Analysis (EDA).
3. Visualized class separability using plots.
4. Trained and compared three classification algorithms:
   - k-NN
   - Logistic Regression
   - Decision Tree
5. Evaluated models using:
   - Accuracy
   - Precision
   - Recall
   - Confusion matrix
6. Saved the best model using `joblib`.
7. Added example inference code.

## Model Results

| Model | Accuracy | Macro Precision | Macro Recall |
| --- | ---: | ---: | ---: |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 |
| k-NN | 0.9667 | 0.9697 | 0.9667 |
| Decision Tree | 0.9333 | 0.9444 | 0.9333 |

The best model is **Logistic Regression**.

## Files

```text
task_1/
  IRIS.csv
  README.md
  iris_classification.ipynb
  iris_classification.py
  iris_models.py
  artifacts/
    iris_best_model.joblib
    model_comparison.csv
    iris_pairplot.png
    petal_scatter.png
    confusion_matrix.png
```

## How To Run

Install the required packages:

```bash
python -m pip install pandas matplotlib seaborn joblib nbformat nbclient ipykernel
```

Run the Python script:

```bash
python iris_classification.py
```

You can also open and run the notebook:

```text
iris_classification.ipynb
```

## Interactive Website

Install the website dependencies and start the Streamlit application:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The website lets users change all four flower measurements, view the predicted
species and class probabilities, and compare the input with the Iris dataset.

## Example Inference

```python
import joblib
import pandas as pd

model = joblib.load("artifacts/iris_best_model.joblib")

sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)

prediction = model.predict(sample)[0]
print(prediction)
```

Output:

```text
Iris-setosa
```

## Deliverables

- Notebook with code and plots
- Saved model file
- README with inference example
