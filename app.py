import os
import pickle
import json
import numpy as np
from flask import Flask, render_template, request

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label mappings
with open("epoch_label_mappings.json", "r") as f:
    mappings = json.load(f)

# Reverse map for stages
reverse_mappings = {k: {int(v2): k2 for k2, v2 in v.items()} for k, v in mappings.items()}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        features = np.array([[float(request.form[name]) for name in [
            "Gender","Age","History","Patient","TakeMedication","Severity",
            "BreathShortness","VisualChanges","NoseBleeding","Whendiagnoused",
            "Systolic","Diastolic","ControlledDiet","FamilyHistory"
        ]]])

        # Make prediction
        prediction = model.predict(features)
        stage = reverse_mappings['Stages'].get(int(prediction[0]), "Unknown")

        color_map = {
            "NORMAL": "green",
            "PRE-HYPERTENSION (Stage-1)": "orange",
            "HYPERTENSION (Stage-2)": "red",
            "HYPERTENSIVE CRISIS (Stage-3)": "darkred"
        }
        color = color_map.get(stage, "black")

        return f"""
        <div style='text-align:center; font-family:Arial; margin-top:50px;'>
            <h2>🩺 Predicted Stage:</h2>
            <h1 style='color:{color};'>{stage}</h1>
        </div>
        """
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port
    app.run(host="0.0.0.0", port=port)
