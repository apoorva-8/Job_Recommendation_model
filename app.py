from flask import Flask, request, jsonify , render_template
import pickle
import os

app = Flask(__name__)

# Load classifier and its vectorizer
with open("rf_classifier_job_recommendation.pkl", "rb") as file:
    rf_classifier = pickle.load(file)

# Load the job recommendation vectorizer
with open("rf_classifier_job_recommendation.pkl", "rb") as file:

    job_vectorizer = pickle.load(file)

@app.route('/')
def home():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Use appropriate vectorizer (you can change this logic as needed)
    vectorized = job_vectorizer.transform([text])
    prediction = rf_classifier.predict(vectorized)

    return jsonify({'prediction': prediction[0]})
