import pdfplumber
import numpy as np
import pandas as pd
import os
import sys

# Ensure we can import helper_functions regardless of where this script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from helper_functions import extrac_feature, get_age, get_e2_d5

def parse_medical_pdf(pdf_path):
    """
    Extracts structured clinical data from a reproductive medicine PDF report.
    Returns a dictionary of patient data.
    """
    # Initialize with None/NaN. 
    # specific keys match the CSV column names exactly.
    data = {
        "patient_id": None,
        "cycle_number": None,
        "Age": None,
        "Protocol": None,
        "AMH": None,
        "n_Follicles": None,
        "E2_day5": None,
        "AFC": np.nan, 
        "Patient Response": None # Standardized to underscore
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
            text = page.extract_text()

            data["patient_id"] = extrac_feature(text, "patient_id")
            data["Protocol"] = extrac_feature(text, "Protocol")
            data["AMH"] = extrac_feature(text, "AMH")
            data["cycle_number"] = extrac_feature(text, "cycle_number")
            data["Age"] = get_age(text)
            data["E2_day5"] = get_e2_d5(text)
                


            page2 = pdf.pages[1]
            text2 = page2.extract_text()
            
            data["n_Follicles"] = extrac_feature(text2, "n_Follicles")
            
            response_p2 = extrac_feature(text2, "Patient Response")
            if response_p2:
                data["Patient Response"] = response_p2

    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None

    return data


def add_patient_to_csv(csv_path, patient_data):
    """
    Adds a new patient dictionary to the CSV.
    """
    if patient_data is None:
        print("No patient data to add.")
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Load or Create CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    new_patient_df = pd.DataFrame([patient_data])

    expected_columns = [
        "patient_id", "cycle_number", "Age", "Protocol", "AMH",
        "n_Follicles", "E2_day5", "AFC", "Patient Response"
    ]
    
    for col in expected_columns:
        if col not in new_patient_df.columns:
            new_patient_df[col] = np.nan

    new_patient_df = new_patient_df[expected_columns]

    df = pd.concat([df, new_patient_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"New patient {patient_data.get('patient_id', 'Unknown')} added successfully!")


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(current_dir)
    
    csv_path = os.path.join(project_root, "data", "raw", "patients.csv")
    
    pdf_path = os.path.join(current_dir, "sample.pdf")

    print(f"Target CSV Path: {csv_path}")
    print(f"Target PDF Path: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    patient_data = parse_medical_pdf(pdf_path)

    print("\nExtracted Data:")
    print(patient_data)

    if patient_data:
        print("\nExtraction Successful! Adding to CSV...")
        add_patient_to_csv(csv_path, patient_data)
    else:
        print("\nExtraction Failed.")

if __name__ == "__main__":
    main()