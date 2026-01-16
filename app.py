from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "mlp_model.pkl"), "rb") as f:
    mlp_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, "selected_features.pkl"), "rb") as f:
    selected_features = pickle.load(f)


# ---------------------------------------------------------
# Home route
# ---------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob = None
    error = None

    if request.method == "POST":
        try:
            # Collect user inputs in correct feature order
            input_values = []
            for feat in selected_features:
                val = request.form.get(feat)
                if val is None or val.strip() == "":
                    raise ValueError(f"Missing value for {feat}")
                input_values.append(float(val))

            # Convert to DataFrame
            X_input = pd.DataFrame([input_values], columns=selected_features)

            # Scale
            X_scaled = scaler.transform(X_input)

            # Predict
            pred = mlp_model.predict(X_scaled)[0]
            proba = mlp_model.predict_proba(X_scaled)[0][1]

            prediction = "Diabetic" if pred == 1 else "Not Diabetic"
            prob = round(float(proba), 3)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        features=selected_features,
        prediction=prediction,
        prob=prob,
        error=error
    )


# ---------------------------------------------------------
# Run app
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
