import pandas as pd
import numpy as np
import warnings


def engineer_features(df):
    """
    Input: Clean DataFrame with ['Age', 'AMH', 'AFC']
    Output: Model-Ready DataFrame with added ['Age_Group_3']
    """
    
    # Create the Age Bins (Logic from your snippet) 
    # We follow our definitions: <35, 35-37, 38-40, >40 (medical standards)
    bins = [0, 35, 37, 40, 100]
    labels = [0, 1, 2, 3] 
    
    
    df['Age_Group_Ordinal'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    

    if 'Age_Group_Ordinal' in df.columns:
        df['Age_Group_3'] = (df['Age_Group_Ordinal'].astype(int) == 3).astype(int)
    else:
        
        df['Age_Group_3'] = 0

    #Clean Up & Formatting
    expected_cols = ['Age', 'AMH', 'AFC', 'Age_Group_3']
    
    if 'Patient_Response' in df.columns:
        return df[expected_cols + ['Patient_Response']]
    else:
        return df[expected_cols]