from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from dotenv import load_dotenv  # ✅ Add this

# ✅ Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and encoders
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("mlb_encoder.pkl", "rb") as f:
    symptom_encoder = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_symptoms = data.get("symptoms", [])

        symptom_vector = symptom_encoder.transform([input_symptoms])
        prediction = model.predict(symptom_vector)
        probabilities = model.predict_proba(symptom_vector)

        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities))

        if confidence < 0.6:
            return jsonify({
                "predicted_disease": "Unknown Disease",
                "confidence": round(confidence, 4)
            }), 200

        return jsonify({
            "predicted_disease": predicted_label,
            "confidence": round(confidence, 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Disease Detection API is running."}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # ✅ Default to 5000 if not set
    app.run(debug=True, host="0.0.0.0", port=port)
