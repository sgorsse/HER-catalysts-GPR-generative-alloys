import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HER Catalyst Predictor", layout="wide")

st.title("🔬 HER Electrocatalyst Predictor")

# Subtitle
st.markdown("""
This application accompanies the publication. It allows for the prediction of **Onset Potential** and **Tafel Slope** for multinary alloys (Ternary to Quinary) using calibrated Gaussian Process Regressor (GPR) models. 
These models were trained on an experimental dataset of 181 entries covering an elemental palette of 18 metals: **Ag, Al, Au, Co, Cr, Cu, Fe, Ir, Mg, Mn, Mo, Ni, Pd, Pt, Rh, Ru, W, and Zn**.
""")

# --- 0. CUSTOM CLASSES (REQUIRED FOR UNPICKLING) ---
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

# --- 1. RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    try:
        m_onset = joblib.load('model_onset.joblib')
        m_tafel = joblib.load('model_tafel.joblib')
        data = pd.read_csv('training_data.csv')
        return m_onset, m_tafel, data
    except FileNotFoundError:
        st.error("Critical Error: Model files (.joblib) or training data (.csv) are missing.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()

model_onset, model_tafel, df_train = load_resources()

# --- 2. PHYSICAL CONSTANTS ---
ATOM_PROPS_STD = {
    'Number': {'Ag':47,'Al':13,'Au':79,'Co':27,'Cr':24,'Cu':29,'Fe':26,'Ir':77,'Mg':12,'Mn':25,'Mo':42,'Ni':28,'Pd':46,'Pt':78,'Rh':45,'Ru':44,'W':74,'Zn':30},
    'Electronegativity': {'Ag':1.93,'Al':1.61,'Au':2.54,'Co':1.88,'Cr':1.66,'Cu':1.90,'Fe':1.83,'Ir':2.20,'Mg':1.31,'Mn':1.55,'Mo':2.16,'Ni':1.91,'Pd':2.20,'Pt':2.28,'Rh':2.28,'Ru':2.20,'W':2.36,'Zn':1.65},
    'AtomicWeight': {'Ag':107.87,'Al':26.98,'Au':196.97,'Co':58.93,'Cr':51.99,'Cu':63.55,'Fe':55.84,'Ir':192.22,'Mg':24.30,'Mn':54.94,'Mo':95.95,'Ni':58.69,'Pd':106.42,'Pt':195.08,'Rh':102.91,'Ru':101.07,'W':183.84,'Zn':65.38},
    'AtomicRadius': {'Ag':1.60,'Al':1.25,'Au':1.36,'Co':1.35,'Cr':1.40,'Cu':1.35,'Fe':1.40,'Ir':1.30,'Mg':1.50,'Mn':1.40,'Mo':1.45,'Ni':1.35,'Pd':1.40,'Pt':1.35,'Rh':1.35,'Ru':1.30,'W':1.35,'Zn':1.35},
}
ALL_ELEMENTS = sorted(list(ATOM_PROPS_STD['Number'].keys()))

# --- 3. FEATURE CALCULATION ENGINE ---
def compute_magpie_features(composition_dict, feature_names):
    feats = {}
    for feat in feature_names:
        parts = feat.split(" ")
        if len(parts) < 3: continue
        stat = parts[1]
        prop = " ".join(parts[2:])
        
        vals = []
        fracs = []
        for el, frac in composition_dict.items():
            if frac > 0:
                val = ATOM_PROPS_STD.get(prop, {}).get(el, 0)
                vals.append(val)
                fracs.append(frac)
        
        vals = np.array(vals)
        fracs = np.array(fracs)
        
        if len(vals) == 0:
            feats[feat] = 0.0
            continue

        if stat == 'mean':
            feats[feat] = np.sum(vals * fracs)
        elif stat == 'minimum':
            feats[feat] = np.min(vals)
        elif stat == 'maximum':
            feats[feat] = np.max(vals)
        elif stat == 'range':
            feats[feat] = np.max(vals) - np.min(vals)
        elif stat == 'mode':
            feats[feat] = vals[np.argmax(fracs)]
        elif stat == 'avg_dev':
            mean_val = np.sum(vals * fracs)
            feats[feat] = np.sum(fracs * np.abs(vals - mean_val))
            
    return pd.DataFrame([feats])

def gpr_predict(model, X_feat):
    inner_model = model.regressor_
    mu_z, sigma_z = inner_model.predict(X_feat, return_std=True)
    scale_factor = model.transformer_.scale_[0]
    mu = model.transformer_.inverse_transform(mu_z.reshape(-1,1)).flatten()[0]
    sigma = sigma_z[0] * scale_factor
    return mu, sigma

# --- 4. INPUT INTERFACE (SMART SLIDERS) ---
st.sidebar.header("Alloy Composition")
n_elems = st.sidebar.number_input("System Size (N Elements)", 3, 5, 3)

selected_elems = []
fractions = []
current_total = 0

# A. Input loop for the first N-1 elements
for i in range(n_elems - 1):
    col1, col2 = st.sidebar.columns([1, 2])
    with col1:
        # Default selection shifted
        default_idx = i if i < len(ALL_ELEMENTS) else 0
        el = st.selectbox(f"Element {i+1}", ALL_ELEMENTS, index=default_idx, key=f"el_{i}")
        selected_elems.append(el)
    with col2:
        # Integer slider
        val = st.slider(f"Atomic % {el}", 0, 100, 0, step=1, key=f"fr_{i}")
        fractions.append(val / 100.0)
        current_total += val

# B. Auto-adjustment for the LAST element
i_last = n_elems - 1
col1, col2 = st.sidebar.columns([1, 2])

with col1:
    default_idx = i_last if i_last < len(ALL_ELEMENTS) else 0
    el_last = st.selectbox(f"Element {n_elems} (Balance)", ALL_ELEMENTS, index=default_idx, key=f"el_{i_last}")
    selected_elems.append(el_last)

with col2:
    # Calculate remainder
    remainder = 100 - current_total
    
    if remainder < 0:
        st.error(f"Total > 100% ({current_total}%). Reduce others.")
        valid_comp = False
        final_val = 0
    else:
        valid_comp = True
        final_val = remainder
    
    # --- FIX: FORCE UPDATE SESSION STATE ---
    # This ensures the visual slider moves even if it's disabled
    key_last = f"fr_{i_last}"
    st.session_state[key_last] = final_val
    
    # Display disabled slider linked to the updated state
    st.slider(f"Atomic % {el_last}", 0, 100, disabled=True, key=key_last)
    fractions.append(final_val / 100.0)

# --- 5. EXECUTION & VISUALIZATION ---
if valid_comp and st.sidebar.button("Run Prediction"):
    
    # 1. Parse Composition
    comp_dict = {el: f for el, f in zip(selected_elems, fractions)}
    formula = "".join([f"{el}{f*100:.0f}" for el, f in comp_dict.items() if f > 0])
    
    # 2. Robust Feature Extraction Strategy
    # -----------------------------------------------------
    feature_ref = None
    
    # Essai 1 : Directement sur le pipeline
    if hasattr(model_onset.regressor_, "feature_names_in_"):
        feature_ref = model_onset.regressor_.feature_names_in_
    
    # Essai 2 : Sur la première étape du pipeline (souvent l'Imputer)
    if feature_ref is None:
        try:
            # On accède à la première étape du pipeline (index 0), qui est le transformer (index 1 du tuple)
            first_step = model_onset.regressor_.steps[0][1]
            if hasattr(first_step, "feature_names_in_"):
                feature_ref = first_step.feature_names_in_
        except Exception:
            pass

    # Essai 3 : Fallback CSV (Dernier recours)
    if feature_ref is None:
        st.warning("⚠️ Impossible de lire les features du modèle. Utilisation du CSV (risque d'erreur).")
        feature_ref = [c for c in df_train.columns if "MagpieData" in c]

    # --- Feature Calculation & Alignment ---
    
    # On calcule ce qu'on peut avec le dictionnaire disponible
    X_input = compute_magpie_features(comp_dict, feature_ref)
    
    # ALIGNEMENT STRICT : On s'assure que X_input a EXACTEMENT les colonnes de feature_ref
    # 1. Ajout des manquantes (remplies par 0.0 si on n'a pas la donnée atomique)
    missing_cols = set(feature_ref) - set(X_input.columns)
    if missing_cols:
        # On utilise un dictionnaire pour l'ajout en masse (plus rapide et évite la fragmentation)
        missing_data = {c: 0.0 for c in missing_cols}
        X_input = pd.concat([X_input, pd.DataFrame(missing_data, index=X_input.index)], axis=1)

    # 2. Suppression des superflues
    extra_cols = set(X_input.columns) - set(feature_ref)
    if extra_cols:
        X_input = X_input.drop(columns=extra_cols)

    # 3. Réorganisation (Crucial)
    X_input = X_input[feature_ref]
    
    # -----------------------------------------------------

    # 3. Inference "Sans Filet" (Numpy Array)
    # On convertit en numpy pour désactiver la vérification des noms de colonnes par sklearn.
    # Comme on a aligné l'ordre juste au-dessus, c'est sûr.
    X_values = X_input.to_numpy()

    try:
        mu_onset, sig_onset = gpr_predict(model_onset, X_values)
        mu_tafel, sig_tafel = gpr_predict(model_tafel, X_values)

        # 4. Results Display
        st.subheader(f"Prediction for: {formula}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Onset Potential", f"{mu_onset:.0f} mV", delta=f"σ = {sig_onset:.0f}", delta_color="off")
        with col2:
            st.metric("Tafel Slope", f"{mu_tafel:.0f} mV/dec", delta=f"σ = {sig_tafel:.0f}", delta_color="off")

        # 5. Pareto Plotting
        st.markdown("### Pareto Front Visualization")
        #  - Triggering logical diagram
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Reference Data
        ax.scatter(df_train['onset_potential'], df_train['tafel_slope'], 
                   c='#d62728', alpha=0.4, label='Experimental Data', s=40, marker='x')
        
        # Predicted Candidate
        ax.errorbar(mu_onset, mu_tafel, xerr=sig_onset, yerr=sig_tafel, 
                    fmt='o', color='#2ca02c', ecolor='black', capsize=4, 
                    markersize=12, markeredgecolor='black', label='Predicted Candidate', zorder=10)
        
        ax.set_xlabel('Onset Potential (mV)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Tafel Slope (mV dec⁻¹)', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True)
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug info - Shape sent to model:", X_values.shape)
