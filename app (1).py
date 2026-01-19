import streamlit as st
import pandas as pd
import joblib

# Load saved model files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# App title
st.title("Customer Churn Prediction App")
st.write("Fill the details below to predict churn")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# When button is clicked
if st.button("Predict Churn"):

    # Convert input into DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "PaymentMethod": [payment],
        "PaperlessBilling": [paperless]
    })

    # One-hot encoding
    input_encoded = pd.get_dummies(input_data, drop_first=True)

    # Match training columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer will stay (Probability: {probability:.2f})")
