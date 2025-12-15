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
# 2. MAGPIE LITE ENGINE
# ==============================================================================
ALL_ELEMS = ['Ag','Al','Au','Co','Cr','Cu','Fe','Ir','Mg','Mn','Mo','Ni','Pd','Pt','Rh','Ru','W','Zn']

def parse_formula(formula):
    """
    Parses string 'Pt90Ni10' to {'Pt': 90.0, 'Ni': 10.0}.
    Returns the raw dictionary and the total sum for validation.
    """
    formula = formula.replace(" ", "")
    matches = re.findall(r'([A-Z][a-z]?)([\d.]*)', formula)
    
    composition = {}
    for el, amt in matches:
        if el not in ALL_ELEMS:
            return None, 0, f"Unsupported element: {el}"
        # Default to 1 if no number is specified (e.g., "PtNi" -> 1, 1)
        amount = float(amt) if amt else 1.0
        composition[el] = amount
    
    total_sum = sum(composition.values())
    
    if total_sum == 0: 
        return None, 0, "Empty composition."
        
    return composition, total_sum, None

def normalize_composition(composition, total_sum):
    """Normalizes raw composition to fractions (sum=1.0)"""
    norm_comp = {}
    for k, v in composition.items():
        norm_comp[k] = v / total_sum
    return norm_comp

def learn_and_patch_physics(df, elements):
    """Recovers atomic properties from training data"""
    mean_cols = [c for c in df.columns if "MagpieData mean" in c]
    props = [c.replace("MagpieData mean ", "") for c in mean_cols]
    
    atom_props = pd.DataFrame(index=elements, columns=props)
    X_comp = df[elements].fillna(0)
    
    for p in props:
        y_p = df[f"MagpieData mean {p}"]
        mask = y_p.notna()
        if mask.sum() > 5:
            lr = LinearRegression(fit_intercept=False).fit(X_comp.loc[mask], y_p.loc[mask])
            atom_props[p] = lr.coef_
        else:
            atom_props[p] = 0.0
    return atom_props, props

def calculate_magpie_lite_single(comp_dict, atom_props_df, props_list, feature_columns_order):
    """Calculates all Magpie statistics"""
    x_vec = np.zeros(len(atom_props_df))
    for el, frac in comp_dict.items():
        if el in atom_props_df.index:
            idx = atom_props_df.index.get_loc(el)
            x_vec[idx] = frac
            
    feats = {}
    for prop in props_list:
        p_vec = atom_props_df[prop].values
        
        # Mean
        mean_val = np.dot(x_vec, p_vec)
        feats[f"MagpieData mean {prop}"] = mean_val
        
        mask = x_vec > 0
        if not any(mask):
            p_present = np.array([0.0])
        else:
            p_present = p_vec[mask]

        feats[f"MagpieData minimum {prop}"] = np.min(p_present)
        feats[f"MagpieData maximum {prop}"] = np.max(p_present)
        feats[f"MagpieData range {prop}"] = np.max(p_present) - np.min(p_present)
        
        idx_mode = np.argmax(x_vec)
        feats[f"MagpieData mode {prop}"] = p_vec[idx_mode]
        feats[f"MagpieData avg_dev {prop}"] = np.sum(x_vec * np.abs(p_vec - mean_val))

    # Align with model expected features
    df_res = pd.DataFrame([feats])
    df_final = pd.DataFrame(columns=feature_columns_order)
    
    for col in feature_columns_order:
        if col in df_res.columns:
            df_final.loc[0, col] = df_res.iloc[0][col]
        else:
            df_final.loc[0, col] = 0.0
            
    return df_final

# ==============================================================================
# 3. ASSETS LOADING
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_data = os.path.join(current_dir, 'data', 'her_catalysts_dataset_v1.csv')
        path_model_onset = os.path.join(current_dir, 'models', 'model_onset_v1.joblib')
        path_model_tafel = os.path.join(current_dir, 'models', 'model_tafel_v1.joblib')

        if not os.path.exists(path_data): return None, None, None, None, None, None
        df_train = pd.read_csv(path_data)
        
        model_onset = joblib.load(path_model_onset)
        model_tafel = joblib.load(path_model_tafel)
        
        atom_props, all_props = learn_and_patch_physics(df_train, ALL_ELEMS)
        
        # Feature name extraction
        try:
            if hasattr(model_onset, 'feature_names_in_'):
                feature_cols = model_onset.feature_names_in_
            elif hasattr(model_onset, 'regressor_'):
                 if hasattr(model_onset.regressor_, 'feature_names_in_'):
                     feature_cols = model_onset.regressor_.feature_names_in_
                 else:
                     feature_cols = [c for c in df_train.columns if c.startswith("MagpieData")]
        except:
             feature_cols = [c for c in df_train.columns if c.startswith("MagpieData")]

        return model_onset, model_tafel, atom_props, all_props, list(feature_cols), df_train
        
    except Exception as e:
        st.error(f"Error loading assets: {str(e)}")
        return None, None, None, None, None, None

model_onset, model_tafel, atom_props, all_props, feature_cols, df_train = load_assets()

# ==============================================================================
# 4. USER INTERFACE
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
    st.markdown("### ⚠️ Input Instructions")
    st.warning("""
    **The sum of concentrations must equal 100.**
    
    ✅ Correct: `Pt90Ni10` (90+10=100)
    ✅ Correct: `Co30Fe20Ni50` (30+20+50=100)
    ❌ Incorrect: `Pt50Ni20` (Sum=70)
    """)
    st.markdown("---")
    st.caption("Powered by 🧠 GPR & ⚛️ Magpie Lite")

# --- Main Input ---
col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Chemical Formula (e.g., Pt90Ni10, Co30Fe20Ni50)", "Pt50Ni50")

# --- Prediction Logic ---
if st.button("🚀 Predict Performance", type="primary"):
    if model_onset is None:
        st.error("Assets not loaded. Please check deployment files.")
    else:
        # 1. Parse and Validate
        raw_comp, total_sum, error = parse_formula(formula_input)
        
        if error:
            st.error(f"Input Error: {error}")
        
        # 2. Check Sum = 100 (with slight tolerance for floats)
        elif not (99.0 <= total_sum <= 101.0):
            st.error(f"⚠️ Composition Error: The sum of concentrations is {total_sum:.1f}, but it must be 100.")
            st.info("Please adjust your formula.")
            
        else:
            # 3. Normalize and Calculate Features
            norm_comp = normalize_composition(raw_comp, total_sum)
            X_input = calculate_magpie_lite_single(norm_comp, atom_props, all_props, feature_cols)
            
            try:
                # 4. Prediction
                pred_onset = model_onset.predict(X_input)[0]
                pred_tafel = model_tafel.predict(X_input)[0]

                st.success(f"Analyzed Composition: {raw_comp}")
                
                # 5. Numerical Results
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("⚡ Onset Potential", f"{pred_onset:.1f} mV", delta="Lower is better", delta_color="inverse")
                with res_col2:
                    st.metric("📉 Tafel Slope", f"{pred_tafel:.1f} mV/dec", delta="Lower is better", delta_color="inverse")
                
                # 6. Visualization (Tafel vs Onset) 
                st.markdown("### 📊 Performance Landscape")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Background: Experimental Data
                ax.scatter(df_train['onset_potential'], df_train['tafel_slope'], 
                           c='lightgray', alpha=0.6, edgecolors='gray', s=50, label='Experimental Data (N=180)')
                
                # Foreground: Prediction
                ax.scatter(pred_onset, pred_tafel, 
                           c='red', marker='*', s=300, edgecolors='black', label=f'Prediction: {formula_input}')
                
                # Labels and Formatting
                ax.set_xlabel("Onset Potential (mV)", fontsize=12)
                ax.set_ylabel("Tafel Slope (mV/dec)", fontsize=12)
                ax.set_title("Predicted Performance vs. Existing Database", fontsize=14)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Invert axes if necessary? 
                # Usually closer to 0 is better for Onset, and lower is better for Tafel.
                # Standard plots usually keep axes increasing.
                
                st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
