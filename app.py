
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Churn Predictor", layout="wide")
st.title("Employee Churn Prediction App")

# Load model
with open("model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Upload file
uploaded_file = st.file_uploader("Upload employee data CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    # Encode categorical columns
    data_encoded = data.copy()
    for col in data_encoded.select_dtypes(include='object').columns:
        if col in label_encoders:
            le = label_encoders[col]
            data_encoded[col] = le.transform(data_encoded[col].astype(str))

    # Predict
    predictions = model.predict(data_encoded)
    data['Churn Prediction'] = predictions

    st.subheader("Prediction Results")
    st.dataframe(data[['employee_id', 'Churn Prediction']])

    # Visualizations
    st.subheader("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='Churn Prediction', data=data, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Average Salary by Churn Prediction")
    fig2, ax2 = plt.subplots()
    sns.barplot(x='Churn Prediction', y='salary', data=data, ax=ax2)
    st.pyplot(fig2)
