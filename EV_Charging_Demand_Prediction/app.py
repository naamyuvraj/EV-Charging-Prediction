import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import preprocess_dataframe
from src.train_model import train_model
from src.predict import make_predictions

st.title("EV Charging Demand Prediction System")

uploaded_file = st.file_uploader("Upload EV Charging Volume CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    df_long = preprocess_dataframe(df)

    model, X_test, y_test = train_model(df_long)

    predictions, mae, rmse = make_predictions(model, X_test, y_test)

    st.subheader("Model Evaluation")
    st.write("MAE:", mae)
    st.write("RMSE:", rmse)

