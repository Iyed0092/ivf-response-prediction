import pandas as pd
import re
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
csv_path = os.path.join(project_root, "data", "raw", "patients.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame() 


def extrac_feature(text, feature):
    """
    Extracts specific medical features using regex.
    """
    r = None  # Default to None if not found

    # Patient ID 
    if feature == "patient_id":
        name_match = re.search(r"Name\s*:\s*(.+)", text, re.IGNORECASE)
        if name_match:
            r = name_match.group(1).strip()

    # Protocol
    if feature == "Protocol":
        protocol_match = re.search(r"Protocol\s*:\s*(.+)", text, re.IGNORECASE)
        if protocol_match:
            r = protocol_match.group(1).strip().lower()

    # AMH
    if feature == "AMH":
        amh_match = re.search(r"AMH\s*:\s*([\d\.]+)", text, re.IGNORECASE)
        if amh_match:
            r = float(amh_match.group(1))

    # Cycle Number
    if feature == "cycle_number":
        cycle_match = re.search(r"Cycle number\s*:\s*(\d+)", text, re.IGNORECASE)
        if cycle_match:
            r = int(cycle_match.group(1))

    # Number of Follicles
    if feature == "n_Follicles":
        follicles_match = re.search(r"Number Of follicles\s*=\s*(\d+)", text, re.IGNORECASE)
        if follicles_match:
            r = int(follicles_match.group(1))  # Usually an integer

    # Patient Response (Target)
    if feature == "Patient Response":
        # Regex handles "has an/a X" with optional parentheses, e.g., "optimal-response (optimal)"
        response_match = re.search(
            r"The patient has (?:an|a)\s*([\w\-]+)(?:\s*\(([\w\-]+)\))?", 
            text, 
            re.IGNORECASE
        )
        if response_match:
            full_response = response_match.group(1)  # e.g., "optimal-response"
            # If parentheses exist, use that; otherwise take first part of hyphenated word
            short_response = response_match.group(2) if response_match.group(2) else full_response.split('-')[0]
            r = short_response.lower()  # e.g., "Optimal"


    return r


def get_line(text, start_with):
    """
    Helper to find the line *before* a line starting with a specific char.
    Specific to the table format in the PDF.
    """
    lines = text.split('\n')
    found_parts = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        # Look for the line that starts with specific char (e.g. "5" for J5 row)
        if line.startswith(start_with):
            if i == 0:
                break 
            
            # The data we want is on the previous line in your specific PDF format
            prev_line = lines[i-1].strip()
            found_parts = prev_line.split()
            break
            
    return found_parts


def get_age(text):
    """
    Calculates age based on Birth Date and Cycle Date found in text.
    """
    try:
        # 1. Extract Cycle Date
        # Assumes date is the first element on the line before the line starting with "1"
        line_parts = get_line(text, "1")
        if not line_parts:
            return None
        monitoring_date_str = line_parts[0]

        # 2. Extract Date of Birth (DOB)
        dob_match = re.search(r"Birth date:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text, re.IGNORECASE)
        
        if dob_match and monitoring_date_str:
            dob_str = dob_match.group(1).strip()
            
            def parse_date_flexible(date_str):
                for fmt in ("%d/%m/%y", "%d-%m-%y", "%d/%m/%Y", "%d-%m-%Y"):
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                raise ValueError(f"Unknown date format: {date_str}")

            dob = parse_date_flexible(dob_str)
            current_cycle_date = parse_date_flexible(monitoring_date_str)

    
            age = current_cycle_date.year - dob.year - (
                (current_cycle_date.month, current_cycle_date.day) < (dob.month, dob.day)
            )
            return int(age)
            
    except Exception as e:
        return None
    return None


def get_e2_d5(text):
    """
    Extracts max numeric value from the line associated with Day 5 ("5").
    """
    parts = get_line(text, "5")
    numeric_lst = []
    
    if not parts:
        return None

    try: 
        for x in parts:
            try:
                clean_x = x.replace(',', '.')
                numeric_lst.append(float(clean_x))
            except ValueError:
                continue

        if numeric_lst:
            return max(numeric_lst)
    except:
        return None
    return None