from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        input_features = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["blood_pressure"]),
            float(request.form["skin_thickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["pedigree"]),
            float(request.form["age"]),
        ]

        # Transform input using the scaler
        input_features = np.array(input_features).reshape(1, -1)
        input_features = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Interpret the result
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return render_template("form.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)