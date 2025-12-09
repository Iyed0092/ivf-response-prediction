import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge


def clean_data(df):
    """
    Ingests the raw dataframe (History + New Patient), performs de-identification, 
    cleaning, imputation (KNN & MICE), and formatting.
    
    Returns: A DataFrame with no missing values, ready for Feature Engineering.
    Columns: ['Age', 'AMH', 'AFC', 'Patient_Response']
    """
    
    df = df.copy() # Protect original data

    #1.DE-IDENTIFICATION 
    if 'patient_id' in df.columns:
        # Create a unique ID for every unique name
        unique_names = df["patient_id"].unique()
        patient_id_dict = {name: idx + 2500 for idx, name in enumerate(unique_names)}
        
        df["patient_id"] = df["patient_id"].map(patient_id_dict)

    #2.HANDLING DUPLICATES & SORTING 
    df["n_missing"] = df.isna().sum(axis=1)
    
    
    # Deduplicate: here the duplicates is the case where the same patient takes the same protocol twice in a year and have the same cycle number
    subset_cols = [c for c in ["patient_id", "cycle_number", "Age", "Protocol"] if c in df.columns]
    df = df.drop_duplicates(subset=subset_cols, keep="first")
    
    # Clean up helper column
    df = df.drop(columns=["n_missing"])
    df = df.reset_index(drop=True)

    #3.correct Protocol column using mapping 
    if 'Protocol' in df.columns:
        correct_protocol = {
            'agonist': 'agonist', 'fixed antagonist': 'fixed antagonist',
            'flexible antagonist': 'flexible antagonist', 'fixed anta': 'fixed antagonist',
            'fix antag': 'fixed antagonist', 'agoni': 'agonist',
            'flex anta': 'flexible antagonist', 'flex antag': 'flexible antagonist'
        }
        df["Protocol"] = df["Protocol"].map(correct_protocol)

    #4.IMPUTE AGE (KNN) : here i considereed that age is related to n_Follicles
    if df['Age'].isna().any():
        knn_cols = [c for c in ["Age", "n_Follicles"] if c in df.columns]
        
        imputer_knn = KNNImputer(n_neighbors=5) 
        df[knn_cols] = imputer_knn.fit_transform(df[knn_cols])

    #5.IMPUTE AMH, AFC, n_Follicles (MICE) : they 're both related to each other (high correlations) so MICE is suitable
    imputation_cols = ['n_Follicles', 'AMH', 'AFC']
    existing_impute_cols = [c for c in imputation_cols if c in df.columns]
    
    if len(existing_impute_cols) > 0:
        mice_imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=10,            
            random_state=42          
        )
        df[existing_impute_cols] = mice_imputer.fit_transform(df[existing_impute_cols])

    #6.TARGET ENCODING (Only if training data) 
    correct_mapping = {'low': 0, 'optimal': 1, 'high': 2, 'Low': 0, 'Optimal': 1, 'High': 2}
    df['Patient_Response'] = df['Patient Response'].map(correct_mapping)
    df.drop(columns=['Patient Response'], inplace=True) 
# Drop rows where 'Patient_Response' is NaN in the original dataframe
    df.dropna(subset=['Patient_Response'], inplace=True)

    #7.CLEANUP FOR CHAMPION MODEL 
    # Return only the columns the model needs
    cols_to_keep = ['Age', 'AMH', 'AFC']
    if 'Patient_Response' in df.columns:
        cols_to_keep.append('Patient_Response')
        
    return df[cols_to_keep]