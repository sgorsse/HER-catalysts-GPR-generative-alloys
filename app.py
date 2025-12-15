import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. CONFIGURATION & CLASSES (Indispensable pour charger le modèle)
# ==============================================================================
st.set_page_config(page_title="HER Catalyst Predictor", layout="wide")

# Cette classe doit être définie AVANT le chargement du pickle
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.to_drop = []
    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self.low_var = df.columns[df.var() == 0]
        corr_matrix = df.drop(columns=self.low_var, errors='ignore').corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    def transform(self, X):
        df = pd.DataFrame(X)
        drop_cols = list(set(self.to_drop) | set(self.low_var))
        return df.drop(columns=drop_cols, errors='ignore').values

# ==============================================================================
# 2. DONNÉES ATOMIQUES COMPLÈTES (MAGPIE)
# ==============================================================================
ALL_ELEMS = ['Ag','Al','Au','Co','Cr','Cu','Fe','Ir','Mg','Mn','Mo','Ni','Pd','Pt','Rh','Ru','W','Zn']

# Base de données complète pour générer TOUTES les features attendues par le pipeline
ATOM_DICT = {
    'Ag': {'Number': 47, 'MendeleevNumber': 65, 'AtomicWeight': 107.868, 'MeltingT': 1234.93, 'Column': 11, 'Row': 5, 'CovalentRadius': 145, 'Electronegativity': 1.93, 'NsValence': 1, 'NpValence': 0, 'NdValence': 10, 'NfValence': 0, 'NValence': 11, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 1, 'GSvolume_pa': 17.07, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Al': {'Number': 13, 'MendeleevNumber': 73, 'AtomicWeight': 26.982, 'MeltingT': 933.47, 'Column': 13, 'Row': 3, 'CovalentRadius': 121, 'Electronegativity': 1.61, 'NsValence': 2, 'NpValence': 1, 'NdValence': 0, 'NfValence': 0, 'NValence': 3, 'NsUnfilled': 0, 'NpUnfilled': 5, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 5, 'GSvolume_pa': 16.51, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Au': {'Number': 79, 'MendeleevNumber': 66, 'AtomicWeight': 196.967, 'MeltingT': 1337.33, 'Column': 11, 'Row': 6, 'CovalentRadius': 136, 'Electronegativity': 2.54, 'NsValence': 1, 'NpValence': 0, 'NdValence': 10, 'NfValence': 14, 'NValence': 11, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 1, 'GSvolume_pa': 16.95, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Co': {'Number': 27, 'MendeleevNumber': 58, 'AtomicWeight': 58.933, 'MeltingT': 1768.00, 'Column': 9, 'Row': 4, 'CovalentRadius': 126, 'Electronegativity': 1.88, 'NsValence': 2, 'NpValence': 0, 'NdValence': 7, 'NfValence': 0, 'NValence': 9, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 3, 'NfUnfilled': 0, 'NUnfilled': 3, 'GSvolume_pa': 11.12, 'GSbandgap': 0.0, 'GSmagmom': 1.6, 'SpaceGroupNumber': 194},
    'Cr': {'Number': 24, 'MendeleevNumber': 49, 'AtomicWeight': 51.996, 'MeltingT': 2180.00, 'Column': 6, 'Row': 4, 'CovalentRadius': 139, 'Electronegativity': 1.66, 'NsValence': 1, 'NpValence': 0, 'NdValence': 5, 'NfValence': 0, 'NValence': 6, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 5, 'NfUnfilled': 0, 'NUnfilled': 6, 'GSvolume_pa': 12.00, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 229},
    'Cu': {'Number': 29, 'MendeleevNumber': 64, 'AtomicWeight': 63.546, 'MeltingT': 1357.77, 'Column': 11, 'Row': 4, 'CovalentRadius': 132, 'Electronegativity': 1.90, 'NsValence': 1, 'NpValence': 0, 'NdValence': 10, 'NfValence': 0, 'NValence': 11, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 1, 'GSvolume_pa': 11.81, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Fe': {'Number': 26, 'MendeleevNumber': 56, 'AtomicWeight': 55.845, 'MeltingT': 1811.00, 'Column': 8, 'Row': 4, 'CovalentRadius': 132, 'Electronegativity': 1.83, 'NsValence': 2, 'NpValence': 0, 'NdValence': 6, 'NfValence': 0, 'NValence': 8, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 4, 'NfUnfilled': 0, 'NUnfilled': 4, 'GSvolume_pa': 11.78, 'GSbandgap': 0.0, 'GSmagmom': 2.2, 'SpaceGroupNumber': 229},
    'Ir': {'Number': 77, 'MendeleevNumber': 60, 'AtomicWeight': 192.217, 'MeltingT': 2719.00, 'Column': 9, 'Row': 6, 'CovalentRadius': 141, 'Electronegativity': 2.20, 'NsValence': 2, 'NpValence': 0, 'NdValence': 7, 'NfValence': 14, 'NValence': 9, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 3, 'NfUnfilled': 0, 'NUnfilled': 3, 'GSvolume_pa': 14.28, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Mg': {'Number': 12, 'MendeleevNumber': 68, 'AtomicWeight': 24.305, 'MeltingT': 923.00, 'Column': 2, 'Row': 3, 'CovalentRadius': 139, 'Electronegativity': 1.31, 'NsValence': 2, 'NpValence': 0, 'NdValence': 0, 'NfValence': 0, 'NValence': 2, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 0, 'GSvolume_pa': 22.89, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 194},
    'Mn': {'Number': 25, 'MendeleevNumber': 50, 'AtomicWeight': 54.938, 'MeltingT': 1519.00, 'Column': 7, 'Row': 4, 'CovalentRadius': 139, 'Electronegativity': 1.55, 'NsValence': 2, 'NpValence': 0, 'NdValence': 5, 'NfValence': 0, 'NValence': 7, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 5, 'NfUnfilled': 0, 'NUnfilled': 5, 'GSvolume_pa': 12.00, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 217},
    'Mo': {'Number': 42, 'MendeleevNumber': 52, 'AtomicWeight': 95.95, 'MeltingT': 2896.00, 'Column': 6, 'Row': 5, 'CovalentRadius': 154, 'Electronegativity': 2.16, 'NsValence': 1, 'NpValence': 0, 'NdValence': 5, 'NfValence': 0, 'NValence': 6, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 5, 'NfUnfilled': 0, 'NUnfilled': 6, 'GSvolume_pa': 15.58, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 229},
    'Ni': {'Number': 28, 'MendeleevNumber': 61, 'AtomicWeight': 58.693, 'MeltingT': 1728.00, 'Column': 10, 'Row': 4, 'CovalentRadius': 124, 'Electronegativity': 1.91, 'NsValence': 2, 'NpValence': 0, 'NdValence': 8, 'NfValence': 0, 'NValence': 10, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 2, 'NfUnfilled': 0, 'NUnfilled': 2, 'GSvolume_pa': 10.94, 'GSbandgap': 0.0, 'GSmagmom': 0.6, 'SpaceGroupNumber': 225},
    'Pd': {'Number': 46, 'MendeleevNumber': 63, 'AtomicWeight': 106.42, 'MeltingT': 1828.05, 'Column': 10, 'Row': 5, 'CovalentRadius': 139, 'Electronegativity': 2.20, 'NsValence': 0, 'NpValence': 0, 'NdValence': 10, 'NfValence': 0, 'NValence': 10, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 0, 'GSvolume_pa': 14.70, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Pt': {'Number': 78, 'MendeleevNumber': 62, 'AtomicWeight': 195.084, 'MeltingT': 2041.40, 'Column': 10, 'Row': 6, 'CovalentRadius': 136, 'Electronegativity': 2.28, 'NsValence': 1, 'NpValence': 0, 'NdValence': 9, 'NfValence': 14, 'NValence': 10, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 1, 'NfUnfilled': 0, 'NUnfilled': 2, 'GSvolume_pa': 15.10, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Rh': {'Number': 45, 'MendeleevNumber': 61, 'AtomicWeight': 102.906, 'MeltingT': 2237.00, 'Column': 9, 'Row': 5, 'CovalentRadius': 142, 'Electronegativity': 2.28, 'NsValence': 1, 'NpValence': 0, 'NdValence': 8, 'NfValence': 0, 'NValence': 9, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 2, 'NfUnfilled': 0, 'NUnfilled': 3, 'GSvolume_pa': 13.78, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 225},
    'Ru': {'Number': 44, 'MendeleevNumber': 59, 'AtomicWeight': 101.07, 'MeltingT': 2607.00, 'Column': 8, 'Row': 5, 'CovalentRadius': 146, 'Electronegativity': 2.20, 'NsValence': 1, 'NpValence': 0, 'NdValence': 7, 'NfValence': 0, 'NValence': 8, 'NsUnfilled': 1, 'NpUnfilled': 0, 'NdUnfilled': 3, 'NfUnfilled': 0, 'NUnfilled': 4, 'GSvolume_pa': 13.56, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 194},
    'W': {'Number': 74, 'MendeleevNumber': 51, 'AtomicWeight': 183.84, 'MeltingT': 3695.00, 'Column': 6, 'Row': 6, 'CovalentRadius': 162, 'Electronegativity': 2.36, 'NsValence': 2, 'NpValence': 0, 'NdValence': 4, 'NfValence': 14, 'NValence': 6, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 6, 'NfUnfilled': 0, 'NUnfilled': 6, 'GSvolume_pa': 15.86, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 229},
    'Zn': {'Number': 30, 'MendeleevNumber': 69, 'AtomicWeight': 65.38, 'MeltingT': 692.68, 'Column': 12, 'Row': 4, 'CovalentRadius': 122, 'Electronegativity': 1.65, 'NsValence': 2, 'NpValence': 0, 'NdValence': 10, 'NfValence': 0, 'NValence': 12, 'NsUnfilled': 0, 'NpUnfilled': 0, 'NdUnfilled': 0, 'NfUnfilled': 0, 'NUnfilled': 0, 'GSvolume_pa': 15.20, 'GSbandgap': 0.0, 'GSmagmom': 0.0, 'SpaceGroupNumber': 194},
}

# ==============================================================================
# 3. MOTEUR MAGPIE LITE & ALIGNEMENT
# ==============================================================================

def parse_formula(formula):
    """Parses 'Pt90Ni10' to {'Pt': 90.0, 'Ni': 10.0}."""
    formula = formula.replace(" ", "")
    matches = re.findall(r'([A-Z][a-z]?)([\d.]*)', formula)
    
    composition = {}
    for el, amt in matches:
        if el not in ALL_ELEMS:
            return None, 0, f"Unsupported element: {el}"
        amount = float(amt) if amt else 1.0
        composition[el] = amount
    
    total_sum = sum(composition.values())
    if total_sum == 0: 
        return None, 0, "Empty composition."
        
    return composition, total_sum, None

def normalize_composition(composition, total_sum):
    norm_comp = {}
    for k, v in composition.items():
        norm_comp[k] = v / total_sum
    return norm_comp

def calculate_all_magpie_features(comp_dict, atom_props_df):
    """Generates ALL standard Magpie stats (mean, min, max, range, mode, avg_dev) for ALL properties."""
    
    x_vec = np.zeros(len(atom_props_df))
    for el, frac in comp_dict.items():
        if el in atom_props_df.index:
            idx = atom_props_df.index.get_loc(el)
            x_vec[idx] = frac
            
    feats = {}
    # Loop over every property (AtomicWeight, MeltingT, etc.)
    for prop in atom_props_df.columns:
        p_vec = atom_props_df[prop].values
        
        # Mean
        mean_val = np.dot(x_vec, p_vec)
        feats[f"MagpieData mean {prop}"] = mean_val
        
        mask = x_vec > 0
        if not any(mask):
            p_present = np.array([0.0])
        else:
            p_present = p_vec[mask]

        # Standard Statistics
        feats[f"MagpieData minimum {prop}"] = np.min(p_present)
        feats[f"MagpieData maximum {prop}"] = np.max(p_present)
        feats[f"MagpieData range {prop}"] = np.max(p_present) - np.min(p_present)
        
        idx_mode = np.argmax(x_vec)
        feats[f"MagpieData mode {prop}"] = p_vec[idx_mode]
        feats[f"MagpieData avg_dev {prop}"] = np.sum(x_vec * np.abs(p_vec - mean_val))

    return pd.DataFrame([feats])

def align_features(df_generated, model_feature_names):
    """
    CRUCIAL STEP: Reindexes the generated DataFrame to match EXACTLY
    the feature names and order expected by the model pipeline.
    Fills missing columns with 0.
    """
    return df_generated.reindex(columns=model_feature_names, fill_value=0)

# ==============================================================================
# 4. CHARGEMENT DES ASSETS
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_data = os.path.join(current_dir, 'data', 'her_catalysts_dataset_v1.csv')
        path_model_onset = os.path.join(current_dir, 'models', 'model_onset_v1.joblib')
        path_model_tafel = os.path.join(current_dir, 'models', 'model_tafel_v1.joblib')

        # Load Models
        model_onset = joblib.load(path_model_onset)
        model_tafel = joblib.load(path_model_tafel)
        
        # Load Training Data for Plot
        if os.path.exists(path_data):
            df_train = pd.read_csv(path_data)
        else:
            df_train = pd.DataFrame(columns=['onset_potential', 'tafel_slope'])
        
        # Load Hardcoded Physics (Robustness)
        atom_props_df = pd.DataFrame.from_dict(ATOM_DICT, orient='index')
        atom_props_df = atom_props_df.reindex(ALL_ELEMS).fillna(0)

        return model_onset, model_tafel, atom_props_df, df_train
        
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None, None

model_onset, model_tafel, atom_props_df, df_train = load_assets()

# ==============================================================================
# 5. INTERFACE UTILISATEUR
# ==============================================================================
st.title("⚗️ HER Catalyst Predictor")
st.markdown("""
**Generative design of electrocatalysts** assisted by AI (Gaussian Process Regression).
Enter a chemical formula to predict its **Onset Potential** and **Tafel Slope**.
""")

with st.sidebar:
    st.header("Settings & Info")
    st.info("AI model calibrated on 180 experimental alloys (Small Data Regime).")
    st.markdown("### Supported Elements")
    st.write(", ".join(ALL_ELEMS))
    st.markdown("---")
    st.warning("""
    **The sum of concentrations must equal 100.**
    ✅ Correct: `Pt90Ni10`
    ❌ Incorrect: `Pt50Ni20` (Sum=70)
    """)

col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Chemical Formula (e.g., Pt90Ni10, Co30Fe20Ni50)", "Pt50Ni50")

if st.button("🚀 Predict Performance", type="primary"):
    if model_onset is None:
        st.error("Assets not loaded.")
    else:
        # 1. Parse
        raw_comp, total_sum, error = parse_formula(formula_input)
        
        if error:
            st.error(f"Input Error: {error}")
        elif not (99.0 <= total_sum <= 101.0):
            st.error(f"⚠️ Composition Error: Sum is {total_sum:.1f}, must be 100.")
            
        else:
            # 2. Normalize
            norm_comp = normalize_composition(raw_comp, total_sum)
            
            # 3. Calculate ALL Features (Generator)
            X_generated = calculate_all_magpie_features(norm_comp, atom_props_df)
            
            try:
                # 4. Align & Predict (The Fix)
                # Check for Onset Model
                if hasattr(model_onset, 'feature_names_in_'):
                    X_onset = align_features(X_generated, model_onset.feature_names_in_)
                    pred_onset = model_onset.predict(X_onset)[0]
                else:
                    st.error("Model version mismatch: 'feature_names_in_' not found.")
                    pred_onset = 0

                # Check for Tafel Model
                if hasattr(model_tafel, 'feature_names_in_'):
                    X_tafel = align_features(X_generated, model_tafel.feature_names_in_)
                    pred_tafel = model_tafel.predict(X_tafel)[0]
                else:
                    pred_tafel = 0

                # 5. Display Results
                st.success(f"Analyzed Composition: {raw_comp}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("⚡ Onset Potential", f"{pred_onset:.1f} mV", delta="Lower is better", delta_color="inverse")
                with res_col2:
                    st.metric("📉 Tafel Slope", f"{pred_tafel:.1f} mV/dec", delta="Lower is better", delta_color="inverse")
                
                # 6. Plot
                if not df_train.empty:
                    st.markdown("### 📊 Performance Landscape")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(df_train['onset_potential'], df_train['tafel_slope'], 
                               c='lightgray', alpha=0.6, edgecolors='gray', s=50, label='Experimental Data')
                    ax.scatter(pred_onset, pred_tafel, 
                               c='red', marker='*', s=300, edgecolors='black', label=f'Prediction: {formula_input}')
                    ax.set_xlabel("Onset Potential (mV)")
                    ax.set_ylabel("Tafel Slope (mV/dec)")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
