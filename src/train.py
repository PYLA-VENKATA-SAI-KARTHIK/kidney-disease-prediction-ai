import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def compute_egfr_mdrd(serum_creatinine, age):
    """
    MDRD formula (Modification of Diet in Renal Disease).
    eGFR = 175 × (Scr)^-1.154 × (Age)^-0.203
    """
    # Handle NaNs by using median if necessary, but here we expect clean input or handled by imputer
    egfr = 175 * (serum_creatinine ** -1.154) * (age ** -0.203)
    return egfr

def get_egfr_tier(egfr):
    if egfr >= 90: return "G1 (Normal)"
    if egfr >= 60: return "G2 (Mild)"
    if egfr >= 45: return "G3a (Mild-Moderate)"
    if egfr >= 30: return "G3b (Moderate-Severe)"
    if egfr >= 15: return "G4 (Severe)"
    return "G5 (Failure)"

def train_models(data_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    # Load dataset
    df = pd.read_csv(data_path)
    
    # Pre-cleaning: remove tabs and convert '?' to NaN
    df = df.replace(to_replace={'\t': '', '\?': np.nan}, regex=True)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # Preprocessing
    # Map categorical to numeric
    df['htn'] = df['htn'].map({'yes': 1, 'no': 0})
    df['dm'] = df['dm'].map({'yes': 1, 'no': 0})
    df['cad'] = df['cad'].map({'yes': 1, 'no': 0})
    df['pe'] = df['pe'].map({'yes': 1, 'no': 0})
    df['ane'] = df['ane'].map({'yes': 1, 'no': 0})
    df['appet'] = df['appet'].map({'good': 1, 'poor': 0})
    df['rbc'] = df['rbc'].map({'normal': 1, 'abnormal': 0})
    df['pc'] = df['pc'].map({'normal': 1, 'abnormal': 0})
    df['pcc'] = df['pcc'].map({'present': 1, 'notpresent': 0})
    df['ba'] = df['ba'].map({'present': 1, 'notpresent': 0})
    
    # Target
    df['classification'] = df['classification'].map({'ckd': 1, 'ckd\t': 1, 'notckd': 0})
    
    # Feature Engineering
    df['eGFR'] = compute_egfr_mdrd(df['sc'], df['age'])
    
    # Define features
    features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'eGFR', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet', 'rbc', 'pc', 'pcc', 'ba']
    
    X = df[features]
    y = df['classification']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    # Train HGB
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_train, y_train)
    
    # Train RF
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Save models and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(hgb, 'models/ckd_hgb_model.joblib')
    joblib.dump(rf, 'models/ckd_rf_model.joblib')
    joblib.dump(imputer, 'models/imputer.joblib')
    joblib.dump(features, 'models/features.joblib')
    
    print("Models trained and saved successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/kidney_disease.csv")
    args = parser.parse_args()
    
    train_models(args.data_path)
