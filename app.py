from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


machine_failure_model = joblib.load("machine_failure_model.pkl")
failure_cause_model = joblib.load("failure_cause_model.pkl")
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


app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predict machine failure and identify failure causes",
    version="1.0.0"
)


class MachineData(BaseModel):
    product_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


@app.get("/")
def home():
    return {
        "message": "Machine Failure Prediction API is running"
    }


@app.post("/predict")
def predict(data: MachineData):

    df = pd.DataFrame([data.model_dump()])

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

    probability = float(
        machine_failure_model.predict_proba(df)[0][1]
    )

    threshold = 0.3

    prediction = 1 if probability >= threshold else 0

    response = {
        "machine_failure": bool(prediction),
        "failure_probability": round(probability, 4),
        "failure_causes": [],
        "recommendations": []
    }

    if prediction == 1:

        X_processed = preprocessor.transform(df)

        causes = failure_cause_model.predict(X_processed)

        cause_names = [
            "Tool Wear Failure",
            "Heat Dissipation Failure",
            "Power Failure",
            "Overstrain Failure",
            "Random Failure"
        ]

        if len(causes.shape) > 1:

            predicted_causes = [
                cause_names[i]
                for i, value in enumerate(causes[0])
                if value == 1
            ]

            response["failure_causes"] = predicted_causes

            response["recommendations"] = [
                recommendations[cause]
                for cause in predicted_causes
            ]

    return response