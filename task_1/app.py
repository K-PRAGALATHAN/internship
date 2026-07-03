from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "IRIS.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "iris_best_model.joblib"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


st.set_page_config(page_title="Iris Model Explorer", page_icon="🌸", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def prediction_probabilities(model, values):
    """Return probabilities from the saved custom logistic-regression model."""
    scaled = model.scaler.transform(values)
    classifier = model.classifier
    logits = scaled @ classifier.weights_ + classifier.bias_
    probabilities = classifier._softmax(logits)[0]
    return pd.DataFrame(
        {"Species": classifier.classes_, "Probability": probabilities}
    ).sort_values("Probability", ascending=False)


data = load_data()
model = load_model()

st.title("Iris Species Classification")
st.write(
    "Change the flower measurements to see how the trained Logistic Regression "
    "model classifies an iris flower."
)

with st.sidebar:
    st.header("Flower measurements")
    values = {}
    labels = {
        "sepal_length": "Sepal length (cm)",
        "sepal_width": "Sepal width (cm)",
        "petal_length": "Petal length (cm)",
        "petal_width": "Petal width (cm)",
    }
    for feature in FEATURES:
        values[feature] = st.slider(
            labels[feature],
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=float(data[feature].median()),
            step=0.1,
        )

sample = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)
prediction = model.predict(sample.to_numpy())[0]
probabilities = prediction_probabilities(model, sample.to_numpy())

result_col, probability_col = st.columns([1, 2])
with result_col:
    st.subheader("Prediction")
    st.success(prediction.replace("Iris-", "").title())
    st.metric("Confidence", f"{probabilities.iloc[0]['Probability']:.1%}")
    st.dataframe(sample.rename(columns=labels), hide_index=True, width="stretch")

with probability_col:
    st.subheader("Class probabilities")
    probability_chart = probabilities.set_index("Species")
    st.bar_chart(probability_chart, y="Probability", horizontal=True)
    st.caption(
        "The model calculates a score for each species and converts the scores "
        "into probabilities. The species with the highest probability is selected."
    )

st.subheader("Your flower compared with the dataset")
fig, ax = plt.subplots(figsize=(9, 5))
colors = {
    "Iris-setosa": "#2563eb",
    "Iris-versicolor": "#16a34a",
    "Iris-virginica": "#dc2626",
}
for species, group in data.groupby("species"):
    ax.scatter(
        group["petal_length"],
        group["petal_width"],
        label=species.replace("Iris-", "").title(),
        color=colors[species],
        alpha=0.65,
    )
ax.scatter(
    values["petal_length"],
    values["petal_width"],
    label="Your input",
    marker="*",
    s=300,
    color="#111827",
    edgecolor="white",
)
ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig)

with st.expander("Model details"):
    st.write(
        "The project compared Logistic Regression, k-Nearest Neighbours, and a "
        "Decision Tree. Logistic Regression was selected because it achieved the "
        "highest test accuracy (100%) on the project's stratified test split."
    )
    comparison = pd.read_csv(BASE_DIR / "artifacts" / "model_comparison.csv")
    st.dataframe(comparison, hide_index=True, width="stretch")
