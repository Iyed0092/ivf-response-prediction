import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from io import BytesIO


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXTRACT_PDF_DIR = os.path.join(PROJECT_ROOT, "extract_pdf")

for path in [SRC_DIR, EXTRACT_PDF_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from extract_pdf import parse_medical_pdf, add_patient_to_csv
    from model.predict import predict_patient_outcome
except ImportError as e:
    st.error(f"🚨 CRITICAL ERROR: Could not import required modules. Error: {e}")
    st.stop()

HISTORY_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "patients.csv")


if 'pdf_data' not in st.session_state:
    st.session_state['pdf_data'] = {}
if 'pdf_uploaded_name' not in st.session_state:
    st.session_state['pdf_uploaded_name'] = ""


@st.cache_data(show_spinner=False)
def run_pdf_extraction(uploaded_file):
    """Handles PDF file object and extracts patient data."""
    # Convert Streamlit UploadedFile to BytesIO for pdfplumber
    pdf_bytes = BytesIO(uploaded_file.getvalue())
    return parse_medical_pdf(pdf_bytes)


st.set_page_config(page_title="IVF Patient Response Predictor", layout="wide")
st.title("🔬 IVF Patient Response Predictor")
st.markdown("Enter patient data manually or upload a medical PDF to auto-fill the form.")

uploaded_pdf = st.sidebar.file_uploader("Upload Medical PDF (Optional)", type=["pdf"])

if uploaded_pdf is not None and uploaded_pdf.name != st.session_state['pdf_uploaded_name']:
    with st.spinner(f"Extracting data from {uploaded_pdf.name}..."):
        extracted_data = run_pdf_extraction(uploaded_pdf)

    if extracted_data:
        st.session_state['pdf_data'] = extracted_data
        st.session_state['pdf_uploaded_name'] = uploaded_pdf.name
        st.sidebar.success("Extraction successful!")
    else:
        st.session_state['pdf_data'] = {}
        st.sidebar.error("Extraction failed. Please check PDF format.")

pdf_data = st.session_state['pdf_data']


with st.form("patient_data_form"):
    st.header("Patient Data Input")

    col1, col2 = st.columns(2)
    patient_id = col1.text_input(
        "Patient Name/ID",
        value=pdf_data.get("patient_id") or "Jane Doe"
    )
    cycle_number = col2.number_input(
        "Cycle Number",
        min_value=1,
        value=int(pdf_data.get("cycle_number") or 1)
    )

    st.subheader("Clinical Markers")
    col3, col4, col5 = st.columns(3)

    # Age
    age_val = pdf_data.get("Age")
    age = col3.number_input(
        "Age (Years)",
        min_value=18, max_value=50,
        value=int(age_val) if age_val is not None and not (isinstance(age_val, float) and np.isnan(age_val)) else 35
    )

    # AMH
    amh_val = pdf_data.get("AMH")
    amh = col4.number_input(
        "AMH (ng/mL)",
        min_value=0.01, max_value=20.0,
        value=float(amh_val) if amh_val is not None and not (isinstance(amh_val, float) and np.isnan(amh_val)) else 1.5,
        step=0.1, format="%.2f"
    )

    # AFC (optional)
    afc_val = pdf_data.get("AFC")
    afc = col5.number_input(
        "AFC (Optional)",
        min_value=0,
        max_value=50,
        value=int(afc_val) if afc_val is not None and not (isinstance(afc_val, float) and np.isnan(afc_val)) else 0,
        help="Leave as 0 if unknown; model/imputation will handle it."
    )

    st.write("")  
    col6, col7 = st.columns(2)
    
    # n_Follicles
    n_follicles_val = pdf_data.get("n_Follicles")
    n_Follicles = col6.number_input(
        "n_Follicles (Required for CSV)",
        min_value=0,
        max_value=200,
        value=int(n_follicles_val) if n_follicles_val is not None and not (isinstance(n_follicles_val, float) and np.isnan(n_follicles_val)) else 0,
        help="Number of follicles. Must be provided to add data to history."
    )

    # Protocol
    protocol_options = ['agonist', 'fixed antagonist', 'flexible antagonist']
    current_protocol = pdf_data.get("Protocol", "flexible antagonist")
    try:
        default_index = protocol_options.index(current_protocol)
    except ValueError:
        default_index = 2
    protocol = st.selectbox(
        "Stimulation Protocol",
        options=protocol_options,
        index=default_index
    )

    st.markdown("---")
    st.subheader("Outcome Data")
    
    response_options = ["Unknown", "Low", "Optimal", "High"]
    
    extracted_response = pdf_data.get("Patient Response") or pdf_data.get("Patient_Response")
    
    if extracted_response and isinstance(extracted_response, str):
        extracted_response = extracted_response.strip().capitalize()
    
    if extracted_response and extracted_response in response_options:
        resp_index = response_options.index(extracted_response)
    else:
        resp_index = 0  
        
    patient_response = st.selectbox(
        "Patient Response (Actual)",
        options=response_options,
        index=resp_index,
        help="Pre-filled from PDF if available. Must be set to Low/Optimal/High to add data to CSV."
    )

    # Buttons
    st.write("")
    col_submit, col_add_data = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button("🔮 Predict Response")
    with col_add_data:
        add_data_button = st.form_submit_button("💾 Add Data to CSV")


def _is_valid_n_follicles(val):
    try:
        return int(val) > 0
    except Exception:
        return False

if submitted:
    if not _is_valid_n_follicles(n_Follicles):
        st.error("Please provide a valid n_Follicles value (> 0). Only AFC may be left blank/0.")
    else:
        input_dict = {
            'patient_id': patient_id,
            'cycle_number': cycle_number,
            'Age': age,
            'AMH': amh,
            'Protocol': protocol,
            'AFC': afc if afc > 0 else None,
            'n_Follicles': int(n_Follicles),
            'Patient_Response': None 
        }

        with st.spinner("Running Prediction Pipeline..."):
            result = predict_patient_outcome(input_dict)

        st.markdown("---")
        st.header("Prediction Results 🩺")
        if isinstance(result, dict) and result.get("Status") == "Success":
            st.success(f"Response Predicted: **{result['Predicted_Response']}**")
            st.metric(
                label="Predicted Ovarian Response",
                value=result['Predicted_Response'],
                delta=f"Confidence: {result['Confidence']}"
            )
            imputed = result.get("Imputed_Values", {})
            st.info(f"Input/Imputed AFC: **{imputed.get('AFC', 'N/A')}**")
        else:
            st.error(f"Prediction failed: {result}")


if add_data_button:
    if not _is_valid_n_follicles(n_Follicles):
        st.error("Cannot add to CSV: please provide a valid n_Follicles value (> 0).")
    elif patient_response == "Unknown":
        st.error("🚨 Cannot add to CSV: Patient Response is missing! Please select 'Low', 'Optimal', or 'High'.")
    else:

        data_to_add = {
            'patient_id': patient_id,
            'cycle_number': cycle_number,
            'Age': age,
            'AMH': amh,
            'Protocol': protocol,
            'AFC': afc if afc > 0 else np.nan,
            'E2_day5': pdf_data.get('E2_day5', np.nan), 
            'n_Follicles': int(n_Follicles),
            'Patient Response': patient_response 
        }

        try:
            with st.spinner("Adding patient data to CSV..."):
                add_patient_to_csv(HISTORY_PATH, data_to_add)
            
            st.success(f"✅ Patient {patient_id} (Outcome: {patient_response}) successfully added to history!")
            
        except Exception as e:
            st.error(f"Failed to update dataset: {e}")