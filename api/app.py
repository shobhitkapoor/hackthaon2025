from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# Load model
model = joblib.load("../model/fixie_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        descriptions = data.get("descriptions", [])
        if not descriptions:
            return jsonify({"error": "Missing 'descriptions' key or value is empty"}), 400

        predictions = model.predict(descriptions)
        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/")
def home():
    return "FixieBot API is running. Use POST /predict"

if __name__ == "__main__":
    app.run(debug=True)
