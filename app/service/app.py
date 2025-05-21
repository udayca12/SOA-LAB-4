from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/sentiment_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return "ML Sentiment Service is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = model.predict([text])[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
