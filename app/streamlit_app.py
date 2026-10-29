import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Estimator", page_icon="🚗")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    with open("options.json") as f:
        options = json.load(f)
    return model, options

model, options = load_artifacts()

st.title("🚗 Used Car Price Estimator")
st.caption("Enter the car's details to get an estimated price.")

col1, col2 = st.columns(2)

with col1:
    year = st.slider("Year", 1990, 2022, 2015)
    manufacturer = st.selectbox("Manufacturer", options["manufacturer"])
    car_model = st.selectbox("Model", options["model"])
    condition = st.selectbox("Condition", options["condition"])
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12], index=1)
    fuel = st.selectbox("Fuel", options["fuel"])

with col2:
    odometer = st.number_input("Odometer (miles)", 0, 400000, 90000, step=1000)
    title_status = st.selectbox("Title status", options["title_status"])
    transmission = st.selectbox("Transmission", options["transmission"])
    drive = st.selectbox("Drive", options["drive"])
    car_type = st.selectbox("Type", options["type"])
    paint_color = st.selectbox("Paint color", options["paint_color"])
    state = st.selectbox("State", options["state"])

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame([{
        "year": float(year),
        "cylinders": float(cylinders),
        "odometer": float(odometer),
        "posting_month": 4,            # near-constant; hardcoded
        "manufacturer": manufacturer,
        "model": car_model,
        "condition": condition,
        "fuel": fuel,
        "title_status": title_status,
        "transmission": transmission,
        "drive": drive,
        "type": car_type,
        "paint_color": paint_color,
        "state": state,
    }])
    pred = np.expm1(model.predict(row)[0])
    st.metric("Estimated price", f"${pred:,.0f}")
    st.caption("Estimate from a RandomForest model. For guidance only.")
