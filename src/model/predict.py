import os
import sys
import pandas as pd
import numpy as np
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing.clean_dataset import clean_data
from preprocessing.feature_engineering import engineer_features

# Paths
HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "patients.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "champion_model_pipeline.joblib")


def predict_patient_outcome(new_patient_dict):

    try:
        if not os.path.exists(HISTORY_PATH):
             # Warning only; create empty DF to prevent crash
             print(f"WARNING: Historical data not found at {HISTORY_PATH}")
             df_history = pd.DataFrame() 
        else:
            df_history = pd.read_csv(HISTORY_PATH)
    except Exception as e:
        return {"Status": "Failure", "Message": f"Error loading history: {str(e)}"}

    # 2. Append New Row & Clean
    new_patient_df = pd.DataFrame([new_patient_dict])
    
    if df_history.empty:
        df_combined = new_patient_df
    else:
        df_combined = pd.concat([df_history, new_patient_df], axis=0, ignore_index=True)

    # 3. Processing Pipeline
    try:
        df_clean = clean_data(df_combined)
        df_features = engineer_features(df_clean)
        
        # Get the last row (the new patient)
        target_row = df_features.iloc[[-1]] 
        
        # Select Features
        model_cols = ['Age', 'AMH', 'AFC', 'Age_Group_3']
        
        # Check for missing columns
        missing = [c for c in model_cols if c not in target_row.columns]
        if missing:
             return {"Status": "Failure", "Message": f"Missing features: {missing}"}
             
        X_new = target_row[model_cols]

    except Exception as e:
        return {"Status": "Failure", "Message": f"Preprocessing Error: {str(e)}"}

    # 4. Prediction
    try:
        if not os.path.exists(MODEL_PATH):
            return {"Status": "Failure", "Message": "Model file not found."}

        full_pipeline = joblib.load(MODEL_PATH)
        
        # Get the predicted Label 
        prediction_label = full_pipeline.predict(X_new)[0]
        
        probabilities = full_pipeline.predict_proba(X_new)[0]
        

        model_classes = full_pipeline.classes_
        pred_index = np.where(model_classes == prediction_label)[0][0]
        
        confidence = probabilities[pred_index]

        if isinstance(prediction_label, (str, np.str_)):
            result_str = str(prediction_label).capitalize()
        else:
            response_map = {0: 'Low', 1: 'Optimal', 2: 'High'}
            result_str = response_map.get(int(prediction_label), "Unknown")

        return {
            "Status": "Success",
            "Predicted_Response": result_str,
            "Confidence": f"{confidence:.2%}",
            "Imputed_Values": {
                "AFC": round(target_row['AFC'].values[0], 2) if 'AFC' in target_row else "N/A"
            }
        }

    except Exception as e:
        return {"Status": "Failure", "Message": f"Prediction Logic Error: {str(e)}"}

if __name__ == "__main__":
    test_input = {
        'patient_id': 'Test', 'cycle_number': 1, 'Age': 32, 
        'AMH': 2.5, 'n_Follicles': 12, 'Protocol': 'agonist', 
        'Patient_Response': None
    }
    print(predict_patient_outcome(test_input))