from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

# Load model and encoders
model = joblib.load("accident_severity_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        start_lat = float(request.form['start_lat'])
        start_lng = float(request.form['start_lng'])
        hour = int(request.form['hour'])
        day_of_week = request.form['day_of_week']
        weather = request.form['weather']
        traffic_signal = int(request.form['traffic_signal'])
        stop = int(request.form['stop'])
        roundabout = int(request.form['roundabout'])

        # Encode categorical features
        day_of_week_encoded = label_encoders['DayOfWeek'].transform([day_of_week])[0]
        weather_encoded = label_encoders['Weather_Condition'].transform([weather])[0]

        # Prepare features array
        features = np.array([[start_lat, start_lng, hour, day_of_week_encoded,
                              weather_encoded, traffic_signal, stop, roundabout]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Map numeric prediction to descriptive label
        severity_map = {1: "Low : Minor damage, no injuries", 2: "Moderate :Noticeable damage, possible minor injuriese : ", 3: "High : Severe damage, possible serious injuries or fatalities"}
        severity_label = severity_map.get(prediction, "Unknown")

        return render_template("index.html",
                               prediction_text=f"Predicted Accident Severity: {severity_label}",
                               severity_label=severity_label)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)