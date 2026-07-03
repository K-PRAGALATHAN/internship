# Machine Learning Model Explorer

An interactive Streamlit website for the three Alfido Tech internship projects:

- Iris species classification
- House price prediction
- Titanic survival prediction

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Commit and push this complete repository to GitHub.
2. Sign in at `https://share.streamlit.io` using GitHub.
3. Select **Create app** and **Yup, I have an app**.
4. Choose the `K-PRAGALATHAN/internship` repository and its main branch.
5. Set the entrypoint file to `app.py`.
6. Open **Advanced settings** and select Python 3.12.
7. Click **Deploy**.

The root `requirements.txt` contains the exact package versions used to create
and load the saved models. No secrets or external system packages are required.

## Technology stack

- Python: application and prediction logic
- Streamlit: website user interface and server
- Pandas and NumPy: input preparation and data processing
- Matplotlib and Streamlit charts: visualizations
- Scikit-learn: house-price and Titanic pipelines
- Joblib: loading the trained model files
