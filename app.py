from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import google.generativeai as genai
import os
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from dotenv import load_dotenv


load_dotenv()
#Secret Key
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY is not set in environment variables")
#Api Key
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY is not set in environment variables")

# Konfigurasi API Key untuk Google Generative AI
genai.configure(api_key=API_KEY)

# Update konfigurasi generasi
generation_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Inisialisasi model Gemini
GEMINI_MODEL_NAME = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    generation_config=generation_config,
)

student_model = tf.keras.models.load_model('./models/nn_student.h5')
professional_model = tf.keras.models.load_model('./models/nn_professional.h5')

student_features = {
    'Gender': ['Male', 'Female'],
    'Age': 'Enter your age (e.g., 25)',
    'Academic Pressure': 'Rate your academic pressure from 1 (low) to 5 (high)',
    'Study Satisfaction': 'Rate your study satisfaction from 1 (low) to 5 (high)',
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Study Hours': 'Enter your daily study hours (e.g., 4)',
    'Financial Stress': 'Rate your financial stress from 1 (low) to 5 (high)',
    'Family History of Mental Illness': ['Yes', 'No']
}

professional_features = {
    'Gender': ['Male', 'Female'],
    'Age': 'Enter your age (e.g., 40)',
    'Work Pressure': 'Rate your work pressure from 1 (low) to 5 (high)',
    'Job Satisfaction': 'Rate your job satisfaction from 1 (low) to 5 (high)',
    'Sleep Duration': ['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'],
    'Dietary Habits': ['Healthy', 'Moderate', 'Unhealthy'],
    'Have you ever had suicidal thoughts ?': ['Yes', 'No'],
    'Work Hours': 'Enter your daily work hours (e.g., 8)',
    'Financial Stress': 'Rate your financial stress from 1 (low) to 5 (high)',
    'Family History of Mental Illness': ['Yes', 'No']
}

app = Flask(__name__)

#JWT Token
def verify_jwt(token):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return decoded, None
    except ExpiredSignatureError:
        return None, "Token has expired"
    except InvalidTokenError:
        return None, "Invalid token"

@app.before_request
def check_token():
    if request.endpoint not in ['home']:  # Tambahkan endpoint tanpa proteksi di sini
        token = request.headers.get('Authorization')
        if not token or not token.startswith("Bearer "):
            return jsonify({'error': 'Invalid or missing token'}), 401
        token = token.split("Bearer ")[1]
        decoded, error = verify_jwt(token)
        if not decoded:
            return jsonify({'error': error}), 401

#Start APP
@app.route('/')
def home():
    return "Welcome to FixU API!"

@app.route('/features/<user_type>', methods=['GET'])
def get_features(user_type):
    if user_type == 'student':
        return jsonify(student_features)
    elif user_type == 'professional':
        return jsonify(professional_features)
    else:
        return jsonify({'error': 'Invalid user type'}), 400

@app.route('/predict/<user_type>', methods=['POST'])
def predict(user_type):
    try:
        if user_type == 'student':
            model = student_model
            features = list(student_features.keys())
        elif user_type == 'professional':
            model = professional_model
            features = list(professional_features.keys())
        else:
            return jsonify({'error': 'Invalid user type'}), 400

        try:
            input_data = {feature: request.json[feature] for feature in features}
        except KeyError as e:
            return jsonify({'error': f'Missing feature: {str(e)}'}), 400

        df_input = pd.DataFrame([input_data])

        # Preprocessing
        for col in df_input.columns:
            if col in ['Gender', 'Dietary Habits', 'Sleep Duration', 'Have you ever had suicidal thoughts ?',
                      'Family History of Mental Illness']:
                df_input[col] = df_input[col].astype('category').cat.codes
            else:
                df_input[col] = df_input[col].astype(float)

        # Make prediction
        probabilities = model.predict(df_input)
        probability = float(probabilities[0]) * 100
        result = "Depression" if probability > 50 else "No Depression"

        # Generate feedback
        feedback = generate_feedback(user_type, input_data, result, probability)

        return jsonify({
            'result': result,
            'probability': probability,
            'feedback': feedback
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

def generate_feedback(user_type, user_input, prediction, probability):
    """
    Menggunakan Google Gemini untuk menghasilkan umpan balik berdasarkan input pengguna, prediksi, dan probabilitas.
    """
    try:
        input_summary = ', '.join([f"{k}: {v}" for k, v in user_input.items()])
        prompt = (
            f"User type: {user_type}\n"
            f"Prediction: {prediction} (Probability: {probability:.2f}%)\n"
            f"User input details: {input_summary}\n"
            "Provide practical and concise feedback based on the prediction and user input. "
            "Avoid lengthy explanations, external resources, or contacts. Ensure the feedback is empathetic but to the point, "
            "focusing on actionable suggestions. Include a simple emoji to create a friendly response at the end of the paragraph."
        )

        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        return f"Error generating feedback: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))