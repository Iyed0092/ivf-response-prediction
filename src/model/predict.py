

import os
import sys
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Directory: {current_dir}")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Assuming the necessary path fix is implemented at the top of your script
from preprocessing.clean_dataset import clean_data
from preprocessing.feature_engineering import engineer_features
# DELETED: from train import get_normalization_pipeline, BINARY_COLS, CONTINUOUS_COLS



# --- CONFIGURATION ---
HISTORY_PATH = r"C:\Users\iyedm\OneDrive\Desktop\Dataset\Dataset\data\raw\patients.csv"
MODEL_PATH = r"C:\Users\iyedm\OneDrive\Desktop\Dataset\Dataset\src\model\champion_model_pipeline.joblib"

def predict_patient_outcome(new_patient_dict):
    """
    Full inference pipeline:
    1. Appends new patient to history (for proper Imputation).
    2. Cleans & Imputes missing values (KNN/MICE).
    3. Adds Features (Age_Group_3).
    4. Predicts (Pipeline handles Normalization automatically).
    """

    # 1. Load Historical Data
    try:
        df_history = pd.read_csv(HISTORY_PATH)
    except FileNotFoundError:
        return "Error: Historical data not found. Cannot perform accurate imputation."

    # 2. Append New Row
    new_patient_df = pd.DataFrame([new_patient_dict])
    df_combined = pd.concat([df_history, new_patient_df], axis=0, ignore_index=True)

    # 3. Run Cleaning & Imputation
    df_clean = clean_data(df_combined)

    # 4. Run Feature Engineering
    df_features = engineer_features(df_clean)

    # 5. Retrieve the New Patient Row (FIXED)
    # the new patient (highest ID) in  the last position.
    target_row = df_features.iloc[[-1]] 
    

    # Select only the columns the model was trained on
    model_cols = ['Age', 'AMH', 'AFC', 'Age_Group_3']
    X_new = target_row[model_cols]

    # 6. Load Model & Predict
    try:
        # This loads the full Pipeline (StandardScaler + SVC)
        full_pipeline = joblib.load(MODEL_PATH)
        
        # .predict() automatically runs the scaler first, then the SVM
        prediction_idx = full_pipeline.predict(X_new)[0]
        prediction_prob = full_pipeline.predict_proba(X_new)[0]
        print(prediction_idx)
        print(prediction_prob)

        # 7. Format Outputw
        response_map = {0: 'Low Response', 1: 'Optimal Response', 2: 'High Response'}
        result = response_map.get(prediction_idx, "Unknown")
        confidence = prediction_prob[prediction_idx]

        return {
            "Status": "Success",
            "Predicted_Response": result,
            "Confidence": f"{confidence:.2%}",
            "Imputed_Values": {
                "AFC": round(target_row['AFC'].values[0], 2),
                "AMH": round(target_row['AMH'].values[0], 2)
            }
        }

    except FileNotFoundError:
        return "Error: Model file not found."
    except Exception as e:
        return f"Prediction Error: {str(e)}"

if __name__ == "__main__":
    # Example Input (Simulating a form submission)
    new_input = {
        'patient_id': 'Amira L',
        'cycle_number': 2,
        'Age': 30,
        'AMH': 3.64,
        'n_Follicles': 9,
        'Protocol': 'flexible antagonist',
        'Patient_Response': None # Unknown
        # 'AFC' is missing, will be imputedd
    }

    print("Running Prediction...")
    result = predict_patient_outcome(new_input)
    print(result)