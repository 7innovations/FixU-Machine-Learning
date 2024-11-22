from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import google.generativeai as genai

API_KEY = "AIzaSyAXeiHd67Fowc0DPfXKWBiOHXmnsUw87x8"
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
GEMINI_MODEL_NAME = "gemini-1.5-flash"
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    generation_config=generation_config,
)

student_model = tf.keras.models.load_model('models/nn_student.h5')
professional_model = tf.keras.models.load_model('models/nn_professional.h5')

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

    try:
        for col in df_input.columns:
            if col in ['Gender', 'Dietary Habits', 'Sleep Duration', 'Have you ever had suicidal thoughts ?',
                       'Family History of Mental Illness']:
                df_input[col] = df_input[col].astype('category').cat.codes
            else:
                df_input[col] = df_input[col].astype(float)
    except Exception as e:
        return jsonify({'error': f'Preprocessing Error: {str(e)}'}), 400

    try:
        probabilities = model.predict(df_input)
        probability = float(probabilities[0]) * 100
        result = "Depression" if probability > 50 else "No Depression"
    except Exception as e:
        return jsonify({'error': f'Model Prediction Error: {str(e)}'}), 500

    feedback = generate_feedback(user_type, input_data, result, probability)

    return jsonify({
        'result': result,
        'probability': probability,
        'feedback': feedback
    })

def generate_feedback(user_type, user_input, prediction, probability):
    """
    Menggunakan Google Gemini untuk menghasilkan umpan balik berdasarkan input pengguna, prediksi, dan probabilitas.
    """
    input_summary = ', '.join([f"{k}: {v}" for k, v in user_input.items()])
    prompt = (
        f"User type: {user_type}\n"
        f"Prediction: {prediction} (Probability: {probability:.2f}%)\n"
        f"User input details: {input_summary}\n"
        "Provide practical and concise feedback based on the prediction and user input. "
        "Avoid lengthy explanations, external resources, or contacts. Ensure the feedback is empathetic but to the point, "
        "focusing on actionable suggestions. Include simple emojis to make the response friendly."
    )

    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message(prompt)

    return response.text

if __name__ == '__main__':
    app.run(debug=True)