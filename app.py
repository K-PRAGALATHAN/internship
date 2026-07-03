from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "task_1"))  # Required by the custom Iris model.

st.set_page_config(page_title="ML Model Explorer", page_icon="📊", layout="wide")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


def iris_page():
    data = load_csv(ROOT / "task_1" / "IRIS.csv")
    model = load_model(ROOT / "task_1" / "artifacts" / "iris_best_model.joblib")
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    st.header("Iris Species Classification")
    st.write("Predict one of three Iris species from four flower measurements.")
    cols = st.columns(4)
    values = []
    for col, feature in zip(cols, features):
        values.append(
            col.slider(
                feature.replace("_", " ").title(),
                float(data[feature].min()),
                float(data[feature].max()),
                float(data[feature].median()),
                0.1,
            )
        )

    sample = np.array([values])
    prediction = model.predict(sample)[0]
    scaled = model.scaler.transform(sample)
    classifier = model.classifier
    probabilities = classifier._softmax(
        scaled @ classifier.weights_ + classifier.bias_
    )[0]
    probability_df = pd.DataFrame(
        {"Species": classifier.classes_, "Probability": probabilities}
    ).set_index("Species")

    result, chart = st.columns([1, 2])
    result.success(f"Prediction: {prediction.replace('Iris-', '').title()}")
    result.metric("Confidence", f"{probabilities.max():.1%}")
    chart.bar_chart(probability_df, horizontal=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    for species, group in data.groupby("species"):
        ax.scatter(group.petal_length, group.petal_width, label=species, alpha=0.55)
    ax.scatter(values[2], values[3], marker="*", s=300, color="black", label="Input")
    ax.set(xlabel="Petal length", ylabel="Petal width")
    ax.legend()
    st.pyplot(fig)


def house_page():
    data = load_csv(ROOT / "task_2" / "data.csv")
    model = load_model(
        ROOT / "task_2" / "artifacts" / "best_house_price_model.joblib"
    )
    st.header("House Price Prediction")
    st.write("Estimate a house price from its property and location details.")

    c1, c2, c3 = st.columns(3)
    bedrooms = c1.number_input("Bedrooms", 0, 10, 3)
    bathrooms = c2.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.5)
    sqft_living = c3.number_input("Living area (sq ft)", 200, 15000, 1800, 100)
    sqft_lot = c1.number_input("Lot area (sq ft)", 500, 1000000, 5000, 500)
    floors = c2.number_input("Floors", 1.0, 4.0, 1.0, 0.5)
    condition = c3.slider("Condition", 1, 5, 3)
    yr_built = c1.number_input("Year built", 1800, 2014, 1995)
    city = c2.selectbox("City", sorted(data["city"].dropna().unique()))
    zipcode = c3.text_input("ZIP code", "98103")
    waterfront = c1.checkbox("Waterfront")
    view = c2.slider("View rating", 0, 4, 0)
    sqft_basement = c3.number_input("Basement area", 0, 5000, 300, 50)
    sqft_above = max(sqft_living - sqft_basement, 0)
    renovated = c1.checkbox("Renovated")
    yr_renovated = (
        c2.number_input("Renovation year", yr_built, 2014, 2010)
        if renovated
        else 0
    )

    sample = pd.DataFrame([{
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "sqft_living": sqft_living, "sqft_lot": sqft_lot, "floors": floors,
        "waterfront": int(waterfront), "view": view, "condition": condition,
        "sqft_above": sqft_above, "sqft_basement": sqft_basement,
        "yr_built": yr_built, "yr_renovated": yr_renovated, "city": city,
        "sale_year": 2014, "sale_month": 6, "zipcode": zipcode,
        "house_age": 2014 - yr_built, "was_renovated": int(renovated),
    }])
    if st.button("Estimate price", type="primary"):
        price = max(float(model.predict(sample)[0]), 0)
        st.success(f"Estimated price: ${price:,.2f}")

    st.image(
        str(ROOT / "task_2" / "artifacts" / "price_vs_sqft_living.png"),
        caption="Relationship between living area and recorded prices",
    )


def titanic_page():
    model = load_model(
        ROOT / "task_3" / "artifacts" / "best_titanic_model.joblib"
    )
    st.header("Titanic Survival Prediction")
    st.write("Estimate survival probability from passenger information.")

    c1, c2, c3 = st.columns(3)
    pclass = c1.selectbox("Passenger class", [1, 2, 3], index=2)
    sex = c2.selectbox("Sex", ["male", "female"])
    age = c3.number_input("Age", 0.1, 100.0, 28.0)
    fare = c1.number_input("Fare", 0.0, 600.0, 7.8958)
    embarked = c2.selectbox("Embarked", ["S", "C", "Q"])
    title = c3.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    sibsp = c1.number_input("Siblings/spouses", 0, 10, 0)
    parch = c2.number_input("Parents/children", 0, 10, 0)
    cabin_present = c3.selectbox("Cabin recorded", ["No", "Yes"])
    family_size = sibsp + parch + 1

    sample = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
        "Parch": parch, "Fare": fare, "Embarked": embarked, "Title": title,
        "FamilySize": family_size, "IsAlone": int(family_size == 1),
        "CabinPresent": cabin_present,
    }])
    probabilities = model.predict_proba(sample)[0]
    prediction = int(np.argmax(probabilities))
    label = "Survived" if prediction else "Not survived"
    st.success(f"Prediction: {label}")
    st.metric("Survival probability", f"{probabilities[1]:.1%}")
    st.bar_chart(
        pd.DataFrame(
            {"Outcome": ["Not survived", "Survived"], "Probability": probabilities}
        ).set_index("Outcome"),
        horizontal=True,
    )

    st.image(
        str(ROOT / "task_3" / "artifacts" / "feature_importance.png"),
        caption="Features that most influenced the selected model",
    )


st.title("Machine Learning Model Explorer")
page = st.sidebar.radio(
    "Choose a model",
    ["Project Overview", "Iris Classification", "House Price", "Titanic Survival"],
)

if page == "Project Overview":
    st.header("Interactive internship projects")
    st.write(
        "Use the navigation menu to enter data, run the saved models, and inspect "
        "their predictions and visualizations."
    )
    st.info(
        "Iris demonstrates multiclass classification, House Price demonstrates "
        "regression, and Titanic demonstrates binary classification."
    )
elif page == "Iris Classification":
    iris_page()
elif page == "House Price":
    house_page()
else:
    titanic_page()

