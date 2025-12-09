import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt 
import seaborn as sns           

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current Directory: {SCRIPT_DIR}")

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from preprocessing.clean_dataset import clean_data
from preprocessing.feature_engineering import engineer_features


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "patients.csv")

MODEL_PATH = os.path.join(SCRIPT_DIR, "champion_model_pipeline.joblib")

print(f"Data Path: {DATA_PATH}")
print(f"Model Path: {MODEL_PATH}")




# Normlalization Pipeline
def get_normalization_pipeline():
    """
    Creates and returns the preprocessing object (ColumnTransformer).
    This encapsulates the exact scaling logic:
    - StandardScale for ['Age', 'AMH', 'AFC']
    - Passthrough for ['Age_Group_3']
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Age', 'AMH', 'AFC']),
            ('cat', 'passthrough', ['Age_Group_3'])
        ]
    )
    return preprocessor

# training function
def train_champion_model():
    """
    Loads processed data, builds the full Pipeline (Normalization + SVM),
    trains the model, and saves it.
    """
    df = pd.read_csv(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    # Separate Features (X) and Target (y)
    X = df[['Age', 'AMH', 'AFC', 'Age_Group_3']]
    y = df['Patient_Response']

    # Split Data (Stratified)
    # CRITICAL: We split BEFORE fitting the scaler to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training on {len(X_train)} samples. Testing on {len(X_test)} samples.")

    # Get Preprocessing Pipeline
    # Call the separate function we created above
    preprocessor = get_normalization_pipeline()

    # Define our Champion Model (SVM)
    champion_svc = SVC(
        C=1, 
        kernel='rbf', 
        gamma=0.01, 
        probability=True, 
        class_weight='balanced',
        random_state=42
    )

    # Create the Full Pipeline
    # This bundles Preprocessing + Model into one object.
    # When we call .fit(), it fits the Scaler AND the SVM.
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', champion_svc)
    ])

    # Train
    print("Training Champion Model...")
    full_pipeline.fit(X_train, y_train)

    # Evaluate 
    print("\n--- Model Evaluation (Test Set) ---")
    y_pred = full_pipeline.predict(X_test)
    y_prob = full_pipeline.predict_proba(X_test)
    
    print(classification_report(y_test, y_pred))
    
    # Calculate weighted AUC
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
        print(f"Weighted ROC AUC: {auc:.4f}")
    except ValueError:
        print("ROC AUC could not be calculated (possibly single class in test set).")



    # --- NEW: Confusion Matrix Visualization ---
    print("\nGenering Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Adjust labels based on your class mapping (e.g., 0=Low, 1=Optimal, 2=High)
    class_names = ['Low', 'Optimal', 'High'] 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.show() # Or use plt.savefig('confusion_matrix.png') to save it
    # -------------------------------------------



    

    # Save the Pipeline
    # This saves the SCALER logic (mean/std from training) AND the SVM weights.
    print(f"Saving model pipeline to {MODEL_PATH}...")
    joblib.dump(full_pipeline, MODEL_PATH)
    print("Done.")

if __name__ == "__main__":
    train_champion_model()
