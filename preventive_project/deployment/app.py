import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Load the trained pipeline from the Hugging Face Model Hub
MODEL_REPO = "treddy333/vehicle-breakdown-preventive-maintenance-model"
MODEL_FILE = "best_preventive_prod_model_v1.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

st.title("Vehicle Engine Preventive Maintenance Prediction")
st.write(
    "Enter current engine sensor readings below. Inputs are assembled into a"
    "**pandas DataFrame** with the same columns used during model training."
)

engine_rpm = st.number_input("Engine RPM", min_value=0.0, max_value=10000.0, value=800.0)
lub_oil_pressure = st.number_input(
    "Lubricating Oil Pressure", min_value=0.0, max_value=20.0, value=3.0
)
fuel_pressure = st.number_input(
    "Fuel Pressure", min_value=0.0, max_value=50.0, value=10.0
)
coolant_pressure = st.number_input(
    "Coolant Pressure", min_value=0.0, max_value=10.0, value=2.0
)
lub_oil_temp = st.number_input(
    "Lubricating Oil Temperature (C)", min_value=-20.0, max_value=200.0, value=80.0
)
coolant_temp = st.number_input(
    "Coolant Temperature (C)", min_value=-20.0, max_value=200.0, value=85.0
)

input_data = pd.DataFrame(
    [
        {
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp,
        }
    ]
)


if st.button("Predict Engine Condition"):
    prediction = int(model.predict(input_data)[0])

    st.subheader("Prediction")
    if prediction == 1:
        st.error("Maintenance Required (Engine Condition = 1)")
    else:
        st.success("Engine Operating Normally (Engine Condition = 0)")
