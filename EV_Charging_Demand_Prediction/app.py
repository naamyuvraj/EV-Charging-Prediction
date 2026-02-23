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

    st.subheader("Model Selection")
    model_type = st.selectbox("Choose a model:", ["Linear Regression", "Random Forest"])

    if st.button("Predict Demand", type="primary"):
        with st.spinner("Running predictions..."):
            model, X_test, y_test = train_model(df_long, model_type)

            predictions, mae, rmse = make_predictions(model, X_test, y_test)

        st.subheader("Model Evaluation")
        st.write("MAE:", mae)
        st.write("RMSE:", rmse)

        st.subheader("Actual vs Predicted Demand")

        fig, ax = plt.subplots()
        ax.plot(y_test.values[:200], label="Actual")
        ax.plot(predictions[:200], label="Predicted")
        ax.legend()

        st.pyplot(fig)


        st.subheader("Monthly Average Demand")

        month_labels = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly = df_long.groupby('month')['volume_kwh'].mean()
        monthly.index = monthly.index.map(month_labels)
        fig3, ax3 = plt.subplots()
        ax3.bar(monthly.index, monthly.values, color='darkorange')
        ax3.set_xlabel("Month")
        ax3.set_ylabel("Average Charging Volume (kWh)")
        ax3.set_title("Monthly Average Demand")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig3)
