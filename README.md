# **FixU Machine Learning**

**FixU Machine Learning** is a deep learning-powered solution designed to detect depression and support mental health initiatives. This repository contains backend, deep learning models, dataset, and notebook.

---

## **Features**  

- **Depression Detection**: Implements supervised learning algorithms to identify signs of depression based on user activity surveys.  
- **Feedback Loop**: Utilizes **Retrieval-Augmented Generation (RAG)** and retrains the **Gemini AI model** with updated data to ensure accurate and context-aware responses.  

---

## **Installation**

To set up the project locally, follow these steps:

### 1. Clone the Repository  
```bash
git clone https://github.com/7Innovations/FixU-Machine-Learning.git
cd FixU-Machine-Learning
```

### 2. Set Up a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## **Docker Setup**

You can also run the application using **Docker** for a containerized and platform-independent environment.

### 1. Build the Docker Image  
```bash
docker build -t fixu-machine-learning .
```

### 2. Run the Docker Container  
```bash
docker run -p 8080:8080 fixu-machine-learning
```

This will start the application and expose it on **port 8080**.

---

## **Folder Structure**  

```plaintext
FixU-Machine-Learning/
├── backend/                    # Backend code for API and model deployment
│   ├── models/                 # Pre-trained and fine-tuned model files
│   │   ├── nn_professional.h5  # Model trained for professional demographics
│   │   ├── nn_student.h5       # Model trained for student demographics
│   ├── app.py                  # API script for handling predictions
│   └── utils.py                # Helper functions for preprocessing and predictions
├── dataset/                    # Datasets used for training and testing
│   ├── depression.csv          # Labeled dataset for depression detection
├── notebooks/                  # Jupyter notebooks for exploratory analysis and model training
│   ├── nn_professional.ipynb   # Training notebook for professional demographic model
│   ├── nn_student.ipynb        # Training notebook for student demographic model
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # Project documentation
└── .gitignore                  # Ignored files and directories
```

---

## **How It Works**

1. **Data Preprocessing**: The application preprocesses raw data from activity surveys, such as removing noise and normalizing input.  
2. **Model Training**: Models are trained separately for different demographics (e.g., students and professionals) to improve prediction accuracy.  
3. **API Integration**: The trained models are deployed using a Flask-based API, making them accessible for prediction requests.  
4. **Continuous Learning**: Feedback data is incorporated via RAG to enhance the model's contextual understanding and response accuracy.  

---

## **Dependencies**

All required dependencies are listed in `requirements.txt`. Key dependencies include:

- **TensorFlow**: For building and training machine learning models.  
- **Flask**: For creating the API to deploy the models.  
- **Pandas**: For data manipulation and analysis.  
- **scikit-learn**: For preprocessing and evaluation metrics.  

---

## **Contributing**

Contributions are welcome! If you'd like to improve this project:  

1. Fork the repository.  
2. Create a feature branch.  
3. Commit your changes.  
4. Submit a pull request.  
