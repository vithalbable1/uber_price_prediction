# uber_price_streamlit.py

import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

# App title
st.title("Uber Price Prediction using XGBoost")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("uber_price_dataset.csv")
    return df

df = load_data()

# Show dataset
if st.checkbox("Show Training Dataset"):
    st.write(df.head())

# Split features and target
X = df.drop("Predicted_Price", axis=1)
y = df["Predicted_Price"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Input fields in sidebar
st.sidebar.header("Enter Input Values")
num_cars = st.sidebar.number_input("Number of Cars Available", min_value=1, max_value=100, value=10)
distance = st.sidebar.slider("Distance to Travel (km)", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
traffic = st.sidebar.slider("Traffic Value (1 = Low, 10 = High)", 1, 10, value=5)

# Prepare input for prediction
input_data = pd.DataFrame([[num_cars, distance, traffic]],
                          columns=["Num_Cars_Available", "Distance_to_Travel", "Traffic_Value"])
input_scaled = scaler.transform(input_data)

# Predict
if st.sidebar.button("Predict Uber Price"):
    price = model.predict(input_scaled)[0]
    st.success(f"Predicted Uber Price: ₹{price:.2f}")
