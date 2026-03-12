# Chronic Kidney Disease (CKD) Prediction Portal 🫁

This project is a machine learning-based diagnostic tool designed to predict the likelihood of Chronic Kidney Disease (CKD) in patients using 25 clinical and laboratory features. It is based on a pipeline originally developed in Google Colab, now converted into a production-ready Streamlit web application.

## 🌟 Project Highlights
*   **Ensemble Modeling:** Combines **HistGradientBoosting** and **Random Forest** classifiers for robust and accurate predictions.
*   **eGFR Integration:** Automated calculation of the Estimated Glomerular Filtration Rate (eGFR) using the MDRD formula, allowing for real-time CKD staging (G1 to G5).
*   **Predictive Diagnostics:** Not only identifies current CKD status but also assesses risks for future conditions like Anemia, Cardiac disease, and ESRD.
*   **Flexible Interface:** Support for both manual patient entry (for doctors/clinicians) and bulk CSV uploads (for batch processing).

## 🛠️ Built With
*   **Python 3.10+**
*   **Streamlit (Frontend Framework)**
*   **Scikit-learn (Machine Learning)**
*   **Pandas & NumPy (Data Manipulation)**
*   **Joblib (Model Serialization)**

## 📂 Project Structure
```text
kidney disease predic/
├── data/               # Placeholder for dataset (kidney_disease.csv)
├── models/             # Saved model files and joblib assets
├── src/                
│   ├── app.py          # Main Streamlit web application
│   └── train.py        # Script for model training and preprocessing
├── README.md           # This file
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed. It's recommended to use a virtual environment.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Training the Model
If the models in `models/` are not present or you want to retrain on new data:
1. Place your `kidney_disease.csv` in the `data/` folder.
2. Run the training script:
```bash
python src/train.py
```

### 4. Running the Web Application
Launch the portal locally using Streamlit:
```bash
streamlit run src/app.py
```

## 🏥 Clinical Foundation
The logic for disease staging and future disease risk is based on clinical guidelines from:
*   **KDIGO (Kidney Disease: Improving Global Outcomes) 2022**
*   **ADA (American Diabetes Association) 2024**
*   **WHO (World Health Organization)**

## 👤 Author
Derived from original Google Colab research by [User].
Converted to professional web platform by GitHub Copilot.
