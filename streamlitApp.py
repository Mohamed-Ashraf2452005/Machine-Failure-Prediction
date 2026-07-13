import streamlit as st
import pandas as pd
import numpy as np
import joblib


failure_model = joblib.load("machine_failure_model.pkl")
cause_model = joblib.load("failure_cause_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


recommendations = {
    "Tool Wear Failure":
        "Replace or inspect the cutting tool and schedule preventive maintenance.",

    "Heat Dissipation Failure":
        "Check cooling systems, airflow, and operating temperature conditions.",

    "Power Failure":
        "Inspect power supply, electrical connections, and machine load.",

    "Overstrain Failure":
        "Reduce machine load and inspect mechanical components for excessive stress.",

    "Random Failure":
        "Perform a full machine inspection to identify unexpected issues."
}


st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Machine Failure Prediction System")

st.info(
    """
    This system predicts:

    • Machine Failure Probability

    • Failure Causes

    • Maintenance Recommendations
    """
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:

    product_type = st.selectbox(
        "Product Type",
        ["L", "M", "H"],
        help="""
        L = Low Quality Product

        M = Medium Quality Product

        H = High Quality Product
        """
    )

    air_temperature = st.number_input(
        "Air Temperature (K)",
        value=298.1,
        help="Ambient air temperature around the machine in Kelvin."
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        value=308.6,
        help="Operating process temperature in Kelvin."
    )

with col2:

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        value=1551,
        help="Machine spindle speed in revolutions per minute."
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=42.8,
        help="Rotational force applied by the machine."
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=0,
        help="Accumulated tool wear time in minutes."
    )

st.markdown("---")

if st.button("🔍 Predict Failure"):

    df = pd.DataFrame([{
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "product_type": product_type
    }])

    # Feature Engineering
    df["Temp_diff"] = (
        df["process_temperature"]
        - df["air_temperature"]
    )

    df["Power_W"] = (
        df["torque"]
        * (df["rotational_speed"] * 2 * np.pi / 60)
    )

    df["Strain"] = (
        df["torque"]
        * df["tool_wear"]
    )

    df["Speed_Torque_ratio"] = (
        df["rotational_speed"]
        / (df["torque"] + 1e-6)
    )

    feature_cols = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
        "Temp_diff",
        "Power_W",
        "Strain",
        "Speed_Torque_ratio",
        "product_type"
    ]

    df = df[feature_cols]

    probability = failure_model.predict_proba(df)[0][1]

    threshold = 0.4

    prediction = 1 if probability >= threshold else 0

    st.markdown("---")

    st.subheader("Prediction Result")

    st.metric(
        label="Failure Probability",
        value=f"{probability:.2%}"
    )

    if prediction == 1:

        st.error("⚠️ Machine Failure Detected")

        X_cause = preprocessor.transform(df)

        causes = cause_model.predict(X_cause)[0]

        cause_names = [
            "Tool Wear Failure",
            "Heat Dissipation Failure",
            "Power Failure",
            "Overstrain Failure",
            "Random Failure"
        ]

        predicted_causes = [
            cause_names[i]
            for i, value in enumerate(causes)
            if value == 1
        ]

        st.subheader("Possible Failure Causes")

        if predicted_causes:

            for cause in predicted_causes:
                st.write("🔹", cause)

            st.subheader("🔧 Maintenance Recommendations")

            for cause in predicted_causes:

                st.success(
                    recommendations[cause]
                )

        else:

            st.warning(
                "No specific cause detected."
            )

    else:

        st.success("✅ Machine is Healthy")
