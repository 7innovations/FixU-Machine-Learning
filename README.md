# **FixU-Machine-Learning**
A machine learning-based solution for depression detection. This repository contains models, training scripts, and other resources to support the mental health detection feature of FixU.
---
## **Features**  
- **Depression Detection**: Using supervised learning to detect depression based on user activity surveys.
- **Feedback**: Utilizing Retrieval-Augmented Generation (RAG) and retraining the Gemini AI model with updated data to provide more accurate and context-aware responses.
---
## **Installation**
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/7Innovations/FixU-Machine-Learning.git
   cd FixU-Machine-Learning
   ```
2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
---
## **Docker Setup**
To run the application using Docker, follow these steps:

1. **Build the Docker image**:
   ```bash
   docker build -t fixu-machine-learning .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8080:8080 fixu-machine-learning
   ```

This will start the application and expose it on port 8080.
---
## **Folder Structure**  
```plaintext
FixU-Machine-Learning/
├── backend/                    # Backend code for API and model deployment
│   ├── models/                 # Contains trained model files
│   │   ├── nn_professional.h5  # Model for professionals
│   │   ├── nn_student.h5       # Model for students
│   ├── app.py                  # API script for predictions
├── dataset/                    # Dataset used for training and testing
│   ├── depression.csv          # Labeled dataset for depression detection
├── notebooks/                  # Jupyter notebooks for model training and analysis
│   ├── nn_professional.ipynb   # Training notebook for professional model
│   ├── nn_student.ipynb        # Training notebook for student model
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
```