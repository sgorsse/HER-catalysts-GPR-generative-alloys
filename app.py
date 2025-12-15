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
# 1. CONFIGURATION & CLASSES
# ==============================================================================
st.set_page_config(page_title="HER Catalyst Predictor", layout="wide")

# Custom class definition (Must be identical to training)
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
# 2. DONNÉES ATOMIQUES (HARDCODED FOR ROBUSTNESS)
# ==============================================================================
ALL_ELEMS = ['Ag','Al','Au','Co','Cr','Cu','Fe','Ir','Mg','Mn','Mo','Ni','Pd','Pt','Rh','Ru','W','Zn']

# Propriétés standard (Magpie/Pymatgen) pour garantir des calculs exacts
ATOM_DICT = {
    'Ag': {'Number': 47, 'MeltingT': 1234.93, 'Electronegativity': 1.93, 'NdValence': 10, 'GSvolume_pa': 17.07, 'CovalentRadius': 145, 'MendeleevNumber': 65, 'NsValence': 1, 'NsUnfilled': 1, 'NsUnfilled': 1, 'NValence': 11},
    'Al': {'Number': 13, 'MeltingT': 933.47, 'Electronegativity': 1.61, 'NdValence': 0, 'GSvolume_pa': 16.51, 'CovalentRadius': 121, 'MendeleevNumber': 73, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 3},
    'Au': {'Number': 79, 'MeltingT': 1337.33, 'Electronegativity': 2.54, 'NdValence': 10, 'GSvolume_pa': 16.95, 'CovalentRadius': 136, 'MendeleevNumber': 66, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 11},
    'Co': {'Number': 27, 'MeltingT': 1768.0, 'Electronegativity': 1.88, 'NdValence': 7, 'GSvolume_pa': 11.12, 'CovalentRadius': 126, 'MendeleevNumber': 58, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 9},
    'Cr': {'Number': 24, 'MeltingT': 2180.0, 'Electronegativity': 1.66, 'NdValence': 5, 'GSvolume_pa': 12.00, 'CovalentRadius': 139, 'MendeleevNumber': 49, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 6},
    'Cu': {'Number': 29, 'MeltingT': 1357.77, 'Electronegativity': 1.90, 'NdValence': 10, 'GSvolume_pa': 11.81, 'CovalentRadius': 132, 'MendeleevNumber': 64, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 11},
    'Fe': {'Number': 26, 'MeltingT': 1811.0, 'Electronegativity': 1.83, 'NdValence': 6, 'GSvolume_pa': 11.78, 'CovalentRadius': 132, 'MendeleevNumber': 56, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 8},
    'Ir': {'Number': 77, 'MeltingT': 2719.0, 'Electronegativity': 2.20, 'NdValence': 7, 'GSvolume_pa': 14.28, 'CovalentRadius': 141, 'MendeleevNumber': 60, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 9},
    'Mg': {'Number': 12, 'MeltingT': 923.0, 'Electronegativity': 1.31, 'NdValence': 0, 'GSvolume_pa': 22.89, 'CovalentRadius': 139, 'MendeleevNumber': 68, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 2},
    'Mn': {'Number': 25, 'MeltingT': 1519.0, 'Electronegativity': 1.55, 'NdValence': 5, 'GSvolume_pa': 12.00, 'CovalentRadius': 139, 'MendeleevNumber': 50, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 7},
    'Mo': {'Number': 42, 'MeltingT': 2896.0, 'Electronegativity': 2.16, 'NdValence': 5, 'GSvolume_pa': 15.58, 'CovalentRadius': 154, 'MendeleevNumber': 50, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 6},
    'Ni': {'Number': 28, 'MeltingT': 1728.0, 'Electronegativity': 1.91, 'NdValence': 8, 'GSvolume_pa': 10.94, 'CovalentRadius': 124, 'MendeleevNumber': 61, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 10},
    'Pd': {'Number': 46, 'MeltingT': 1828.05, 'Electronegativity': 2.20, 'NdValence': 10, 'GSvolume_pa': 14.70, 'CovalentRadius': 139, 'MendeleevNumber': 63, 'NsValence': 0, 'NsUnfilled': 0, 'NValence': 10},
    'Pt': {'Number': 78, 'MeltingT': 2041.4, 'Electronegativity': 2.28, 'NdValence': 9, 'GSvolume_pa': 15.10, 'CovalentRadius': 136, 'MendeleevNumber': 62, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 10},
    'Rh': {'Number': 45, 'MeltingT': 2237.0, 'Electronegativity': 2.28, 'NdValence': 8, 'GSvolume_pa': 13.78, 'CovalentRadius': 142, 'MendeleevNumber': 61, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 9},
    'Ru': {'Number': 44, 'MeltingT': 2607.0, 'Electronegativity': 2.20, 'NdValence': 7, 'GSvolume_pa': 13.56, 'CovalentRadius': 146, 'MendeleevNumber': 59, 'NsValence': 1, 'NsUnfilled': 1, 'NValence': 8},
    'W': {'Number': 74, 'MeltingT': 3695.0, 'Electronegativity': 2.36, 'NdValence': 4, 'GSvolume_pa': 15.86, 'CovalentRadius': 162, 'MendeleevNumber': 51, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 6},
    'Zn': {'Number': 30, 'MeltingT': 692.68, 'Electronegativity': 1.65, 'NdValence': 10, 'GSvolume_pa': 15.20, 'CovalentRadius': 122, 'MendeleevNumber': 69, 'NsValence': 2, 'NsUnfilled': 0, 'NValence': 12},
}

# Features EXACTES attendues par chaque modèle (selon votre demande)
FEATURES_ONSET = [
   "MagpieData maximum Number",
   "MagpieData mean Number",
   "MagpieData mode Number",
   "MagpieData maximum MeltingT",
   "MagpieData range MeltingT",
   "MagpieData range Electronegativity",
   "MagpieData mode NdValence",
   "MagpieData maximum GSvolume_pa"
]

FEATURES_TAFEL = [
   "MagpieData minimum Number",
   "MagpieData maximum Number",
   "MagpieData avg_dev Number",
   "MagpieData mean MendeleevNumber",
   "MagpieData range MeltingT",
   "MagpieData range CovalentRadius",
   "MagpieData minimum Electronegativity",
   "MagpieData range Electronegativity",
   "MagpieData mean NsValence",
   "MagpieData avg_dev NsValence",
   "MagpieData mean NsUnfilled",
   "MagpieData maximum GSvolume_pa"
]

# ==============================================================================
# 3. MOTEUR MAGPIE LITE
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

def get_atomic_props():
    """Returns the hardcoded atomic properties dataframe."""
    df = pd.DataFrame.from_dict(ATOM_DICT, orient='index')
    # Ensure all elements are present
    df = df.reindex(ALL_ELEMS).fillna(0)
    return df, df.columns.tolist()

def calculate_magpie_lite_single(comp_dict, atom_props_df, props_list, feature_columns_order):
    """Calculates Magpie statistics for the requested feature columns."""
    
    # Create composition vector
    x_vec = np.zeros(len(atom_props_df))
    for el, frac in comp_dict.items():
        if el in atom_props_df.index:
            idx = atom_props_df.index.get_loc(el)
            x_vec[idx] = frac
            
    feats = {}
    # Iterate over all available physical properties (Number, MeltingT, etc.)
    for prop in props_list:
        if prop not in atom_props_df.columns:
            continue
            
        p_vec = atom_props_df[prop].values
        
        # Mean
        mean_val = np.dot(x_vec, p_vec)
        feats[f"MagpieData mean {prop}"] = mean_val
        
        # Filter for present elements
        mask = x_vec > 0
        if not any(mask):
            p_present = np.array([0.0])
            weights_present = np.array([0.0])
        else:
            p_present = p_vec[mask]
            weights_present = x_vec[mask]

        # Min/Max/Range
        feats[f"MagpieData minimum {prop}"] = np.min(p_present)
        feats[f"MagpieData maximum {prop}"] = np.max(p_present)
        feats[f"MagpieData range {prop}"] = np.max(p_present) - np.min(p_present)
        
        # Mode (property of the element with highest proportion)
        idx_mode = np.argmax(x_vec)
        feats[f"MagpieData mode {prop}"] = p_vec[idx_mode]
        
        # Avg Dev
        feats[f"MagpieData avg_dev {prop}"] = np.sum(x_vec * np.abs(p_vec - mean_val))

    # Construct the final DataFrame strictly in the order requested by the model
    df_final = pd.DataFrame(index=[0])
    
    for col in feature_columns_order:
        if col in feats:
            df_final[col] = feats[col]
        else:
            # Fallback if a feature is missing (should not happen with hardcoded props)
            df_final[col] = 0.0
            
    return df_final[feature_columns_order]

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

        if not os.path.exists(path_data): return None, None, None, None, None
        df_train = pd.read_csv(path_data)
        
        model_onset = joblib.load(path_model_onset)
        model_tafel = joblib.load(path_model_tafel)
        
        # Load Hardcoded Physics
        atom_props, all_props = get_atomic_props()

        return model_onset, model_tafel, atom_props, all_props, df_train
        
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None, None, None

model_onset, model_tafel, atom_props, all_props, df_train = load_assets()

# ==============================================================================
# 5. INTERFACE UTILISATEUR
# ==============================================================================
st.title("⚗️ HER Catalyst Predictor")
st.markdown("""
**Generative design of electrocatalysts** assisted by AI (Gaussian Process Regression).
Enter a chemical formula to predict its **Onset Potential** and **Tafel Slope**.
""")

# --- Sidebar ---
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

# --- Main Input ---
col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Chemical Formula (e.g., Pt90Ni10, Co30Fe20Ni50)", "Pt50Ni50")

# --- Prediction Logic ---
if st.button("🚀 Predict Performance", type="primary"):
    if model_onset is None:
        st.error("Assets not loaded. Please check deployment files.")
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
            
            # 3. Calculate Features (SPECIFIC TO EACH MODEL)
            # Onset Input
            X_onset = calculate_magpie_lite_single(norm_comp, atom_props, all_props, FEATURES_ONSET)
            # Tafel Input
            X_tafel = calculate_magpie_lite_single(norm_comp, atom_props, all_props, FEATURES_TAFEL)
            
            try:
                # 4. Prediction
                pred_onset = model_onset.predict(X_onset)[0]
                pred_tafel = model_tafel.predict(X_tafel)[0]

                st.success(f"Analyzed Composition: {raw_comp}")
                
                # 5. Results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("⚡ Onset Potential", f"{pred_onset:.1f} mV", delta="Lower is better", delta_color="inverse")
                with res_col2:
                    st.metric("📉 Tafel Slope", f"{pred_tafel:.1f} mV/dec", delta="Lower is better", delta_color="inverse")
                
                # 6. Plot
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
