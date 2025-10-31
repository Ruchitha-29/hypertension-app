from flask import Flask, render_template, request
import numpy as np
import pickle
import json

# Load model
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
        gender = float(request.form['Gender'])
        age = float(request.form['Age'])
        history = float(request.form['History'])
        patient = float(request.form['Patient'])
        take_med = float(request.form['TakeMedication'])
        severity = float(request.form['Severity'])
        breath = float(request.form['BreathShortness'])
        visual = float(request.form['VisualChanges'])
        nose = float(request.form['NoseBleeding'])
        whendiag = float(request.form['Whendiagnoused'])
        systolic = float(request.form['Systolic'])
        diastolic = float(request.form['Diastolic'])
        controldiet = float(request.form['ControlledDiet'])
        family = float(request.form['FamilyHistory'])

        features = np.array([[gender, age, history, patient, take_med, severity,
                              breath, visual, nose, whendiag, systolic, diastolic,
                              controldiet, family]])

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
            <a href="/" style='display:inline-block; margin-top:20px;'>🔙 Predict Again</a>
        </div>
        """
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
