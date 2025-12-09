# IVF Response Prediction: AI for Safer Ovarian Stimulation

## 🏥 Project Overview
**In Vitro Fertilization (IVF)** protocols involve stimulating the ovaries to produce multiple eggs. However, this process carries a risk of **Ovarian Hyperstimulation Syndrome (OHSS)**, a potentially life-threatening complication.

This project utilizes machine learning to predict patient response to stimulation protocols (**Low**, **Optimal**, or **High/Risk**). The primary goal is **Clinical Safety**: developing a "Safety-First" model that prioritizes the detection of High Responders to prevent OHSS, ensuring patients receive personalized and safe treatment plans.

---

## 📂 Project Structure

* **`src/preprocessing/clean_dataset.py`**: Handles all initial data cleaning tasks (deduplication, protocol standardization) to ensure raw data integrity.
* **`notebooks/Workshop.ipynb`**: The central experimentation hub. This notebook contains:
    * **Feature Engineering:** Comprehensive selection process using both statistical methods (Pearson Correlation, Hypothesis Testing/ANOVA) and model-based selection (Decision Trees, Linear Models).
    * **Model Selection:** Grid Search comparisons of SVM, Random Forest, XGBoost, and Logistic Regression to identify the Champion Model.
    * **Explainable AI (XAI):** Full implementation of SHAP and LIME to interpret model decisions.
* **`src/model/train.py`**: The production-ready training script that builds and saves the final Champion Pipeline.

---

## 🚀 Key Features
* **Safety-First Classification:** Optimized decision thresholds to prioritize **Recall** for the "High Response" class, acting as an early warning system for OHSS.
* **Rigorous Preprocessing:**
    * **MICE Imputation:** Handled missing biomarkers (AFC/AMH) using correlations.
    * **Biologically-Aware Feature Engineering:** Discretized Age into clinically relevant groups (e.g., >40 cliff).
    * **Custom Normalization:** Applied Square Root transformations to skewed features (AFC) before scaling.
* **Explainable AI (XAI):**
    * **SHAP Analysis:** Validated that the model relies on physiological markers (AFC, AMH) rather than bias.
    * **LIME:** Provided instance-level "sanity checks" for individual patient predictions.
    * **Feature Importance:** Confirmed the biological hierarchy: `AFC > AMH > Age`.

---

## 📊 Methodology

### 1. Data Cleaning (`clean_dataset.py`)
* **Standardization:** Mapped inconsistent protocol names (e.g., typos) to three distinct categories: Agonist, Fixed Antagonist, Flexible Antagonist.
* **Deduplication:** Removed duplicate patient cycles to prevent data leakage.
* **De-identification:** Anonymized patient IDs for privacy compliance.

### 2. Feature Selection (`Workshop.ipynb`)
We employed a multi-stage selection process to identify the "Gold Standard" subset:
* **Statistical Selection:** Used Pearson Correlation to remove noisy features (e.g., `E2_day5`) and Hypothesis Testing (ANOVA) to validate group differences.
* **Model-Based Selection:** Leveraged feature importance from Tree-based models (Random Forest) and coefficient shrinkage from Linear Models (Lasso/Ridge) to finalize the feature set.
* **Selected Features:** `['Age', 'AMH', 'AFC', 'Age_Group_3']`
    * *Result:* Reduced dimensions by 50% while maintaining >93% AUC.

### 3. Model Selection
Comparison of **SVM**, **Random Forest**, **XGBoost**, and **Logistic Regression**.
* **Champion Model:** **Support Vector Machine (SVM)** (RBF Kernel).
* **Performance:**
    * **Weighted ROC AUC:** **0.9332**
    * **High Response Recall:** Optimized to **>0.90** (at threshold 0.24).

---

## 🧠 Explainability & Clinical Validation

### Global Logic (SHAP)
The SHAP summary plot confirms the model follows biological principles:
* **High AFC/AMH** are the strongest drivers for "High Response" risk.
* **Youth (<35)** is a risk factor, while **Age >40** acts as a protective "brake" against hyperstimulation.

### Local Logic (LIME)
LIME analysis of individual cases (e.g., Patient #5) demonstrated the model's nuance:
* *Scenario:* A young patient (high risk demographic) with average AFC (low risk biomarker).
* *Prediction:* **Optimal** (Correct).
* *Reasoning:* The model correctly prioritized the **biomarkers over the age**, avoiding a False Positive.

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `lime`

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Iyed0092/ivf-response-prediction.git](https://github.com/Iyed0092/ivf-response-prediction.git)
    cd ivf-response-prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Training Pipeline
To retrain the Champion Model and generate the pipeline file:
```bash
python src/model/train.py