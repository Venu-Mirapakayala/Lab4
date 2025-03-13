from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained classification model
model = joblib.load("fish_classification_model.pkl")

# Load label encoder to map numbers back to species names
label_encoder = joblib.load("species_label_encoder.pkl")

@app.route("/")
def home():
    return render_template("page.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract user input from form
        form_data = request.form.to_dict()

        # Convert input values to float
        input_features = [float(value) for value in form_data.values()]

        # Make prediction
        prediction_numeric = model.predict([input_features])[0]

        # Convert numeric prediction back to actual species name
        predicted_species = label_encoder.inverse_transform([prediction_numeric])[0]

        return render_template("page.html",prediction = predicted_species)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
