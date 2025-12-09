import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. CONFIGURATION & CLASSES
# ==============================================================================
st.set_page_config(page_title="HER Catalyst Predictor", layout="wide")

# Définition de la classe personnalisée (Indispensable pour charger le modèle)
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
# 2. MOTEUR "MAGPIE LITE" (Embarqué)
# ==============================================================================
ALL_ELEMS = ['Ag','Al','Au','Co','Cr','Cu','Fe','Ir','Mg','Mn','Mo','Ni','Pd','Pt','Rh','Ru','W','Zn']

def parse_formula(formula):
    """Convertit 'Pt90Ni10' en {'Pt':0.9, 'Ni':0.1}"""
    formula = formula.replace(" ", "")
    matches = re.findall(r'([A-Z][a-z]?)([\d.]*)', formula)
    
    composition = {}
    for el, amt in matches:
        if el not in ALL_ELEMS:
            return None, f"Élément non supporté : {el}"
        amount = float(amt) if amt else 1.0
        composition[el] = amount
    
    total = sum(composition.values())
    if total == 0: return None, "Composition vide"
    for k in composition:
        composition[k] /= total
        
    return composition, None

def learn_and_patch_physics(df, elements):
    """Reconstruit la table des propriétés atomiques depuis les données d'entraînement"""
    mean_cols = [c for c in df.columns if "MagpieData mean" in c]
    props = [c.replace("MagpieData mean ", "") for c in mean_cols]
    atom_props = pd.DataFrame(index=elements, columns=props)
    X_comp = df[elements].fillna(0)
    
    for p in props:
        y_p = df[f"MagpieData mean {p}"]
        mask = y_p.notna()
        if mask.sum() > 10:
            lr = LinearRegression(fit_intercept=False).fit(X_comp.loc[mask], y_p.loc[mask])
            atom_props[p] = lr.coef_
        else:
            atom_props[p] = 0.0

    std_vals = {
        'Number': {'Ag':47,'Al':13,'Au':79,'Co':27,'Cr':24,'Cu':29,'Fe':26,'Ir':77,'Mg':12,'Mn':25,'Mo':42,'Ni':28,'Pd':46,'Pt':78,'Rh':45,'Ru':44,'W':74,'Zn':30},
        'Electronegativity': {'Ag':1.93,'Al':1.61,'Au':2.54,'Co':1.88,'Cr':1.66,'Cu':1.90,'Fe':1.83,'Ir':2.20,'Mg':1.31,'Mn':1.55,'Mo':2.16,'Ni':1.91,'Pd':2.20,'Pt':2.28,'Rh':2.28,'Ru':2.20,'W':2.36,'Zn':1.65}
    }
    for col, data in std_vals.items():
        if col in atom_props.columns:
            for el, val in data.items():
                if el in atom_props.index: atom_props.loc[el, col] = val
                
    return atom_props, props

def calculate_magpie_lite_single(comp_dict, atom_props_df, props_list, feature_columns_order):
    """Calcule les features pour une seule composition"""
    x_vec = np.zeros(len(atom_props_df))
    for el, frac in comp_dict.items():
        if el in atom_props_df.index:
            idx = atom_props_df.index.get_loc(el)
            x_vec[idx] = frac
            
    feats = {}
    for prop in props_list:
        p_vec = atom_props_df[prop].values
        mean_val = np.dot(x_vec, p_vec)
        feats[f"MagpieData mean {prop}"] = mean_val
        
        present_mask = x_vec > 0
        if not any(present_mask):
            p_present = [0]
        else:
            p_present = p_vec[present_mask]
            
        feats[f"MagpieData minimum {prop}"] = np.min(p_present)
        feats[f"MagpieData maximum {prop}"] = np.max(p_present)
        feats[f"MagpieData range {prop}"] = np.max(p_present) - np.min(p_present)
        
        idx_max = np.argmax(x_vec)
        feats[f"MagpieData mode {prop}"] = p_vec[idx_max]
        feats[f"MagpieData avg_dev {prop}"] = np.sum(x_vec * np.abs(p_vec - mean_val))

    df_res = pd.DataFrame([feats])
    for col in feature_columns_order:
        if col not in df_res.columns:
            df_res[col] = 0.0
            
    return df_res[feature_columns_order]

# ==============================================================================
# 3. CHARGEMENT DES RESSOURCES (Mis à jour pour la structure de dossiers)
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
        # Récupération du chemin absolu du dossier où se trouve app.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construction des chemins vers les sous-dossiers
        path_data = os.path.join(current_dir, 'data', 'training_data.csv')
        path_model_onset = os.path.join(current_dir, 'models', 'model_onset.joblib')
        path_model_tafel = os.path.join(current_dir, 'models', 'model_tafel.joblib')

        # 1. Charger les données
        if not os.path.exists(path_data):
            st.error(f"Fichier introuvable : {path_data}")
            return None, None, None, None, None, None
            
        df_train = pd.read_csv(path_data)
        
        # 2. Charger les modèles
        if not os.path.exists(path_model_onset) or not os.path.exists(path_model_tafel):
            st.error("Fichiers modèles introuvables dans le dossier 'models/'.")
            return None, None, None, None, None, None

        model_onset = joblib.load(path_model_onset)
        model_tafel = joblib.load(path_model_tafel)
        
        # 3. Calibration
        atom_props, all_props = learn_and_patch_physics(df_train, ALL_ELEMS)
        
        # 4. Ordre des colonnes
        feature_cols = [c for c in df_train.columns if c.startswith("MagpieData")]
        
        return model_onset, model_tafel, atom_props, all_props, feature_cols, df_train
        
    except Exception as e:
        st.error(f"Erreur critique lors du chargement : {str(e)}")
        return None, None, None, None, None, None

model_onset, model_tafel, atom_props, all_props, feature_cols, df_train = load_assets()

# ==============================================================================
# 4. INTERFACE UTILISATEUR
# ==============================================================================
st.title("⚗️ High-Performance HER Catalyst Predictor")
st.markdown("""
**Conception générative d'électrocatalyseurs** assistée par l'IA (Gaussian Process Regression).
Entrez une formule chimique pour prédire son **Potentiel d'Onset** et sa **Pente de Tafel**.
""")

with st.sidebar:
    st.header("Paramètres")
    st.info("Modèle calibré sur 180 alliages expérimentaux (Small Data Regime).")
    st.markdown("### Éléments Supportés")
    st.write(", ".join(ALL_ELEMS))
    st.markdown("---")
    st.caption("Développé avec 🧠 GPR & ⚛️ Magpie Lite")

col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Formule Chimique (ex: Pt90Ni10, Co30Fe20Ni50)", "Pt50Ni50")

if st.button("🚀 Prédire la Performance", type="primary"):
    if model_onset is None:
        st.error("Impossible de faire une prédiction : les ressources n'ont pas été chargées.")
    else:
        comp_dict, error = parse_formula(formula_input)
        
        if error:
            st.warning(error)
        else:
            X_input = calculate_magpie_lite_single(comp_dict, atom_props, all_props, feature_cols)
            
            try:
                pred_onset = model_onset.predict(X_input)[0]
                pred_tafel = model_tafel.predict(X_input)[0]

                st.success(f"Composition analysée : {comp_dict}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(
                        label="⚡ Onset Potential",
                        value=f"{pred_onset:.1f} mV",
                        delta="Plus bas est mieux",
                        delta_color="inverse"
                    )
                with res_col2:
                    st.metric(
                        label="📉 Tafel Slope",
                        value=f"{pred_tafel:.1f} mV/dec",
                        delta="Plus bas est mieux",
                        delta_color="inverse"
                    )
                
                score = (pred_onset + pred_tafel) / 2
                if score < 40:
                    st.balloons()
                    st.info("🌟 Candidat Exceptionnel !")
                elif score < 60:
                    st.info("✅ Bon candidat.")
                else:
                    st.warning("⚠️ Performance modeste attendue.")
                    
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
