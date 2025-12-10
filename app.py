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
# 2. MOTEUR "MAGPIE LITE" (ROBUSTE)
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
    # On cherche toutes les propriétés disponibles via la moyenne ("mean")
    mean_cols = [c for c in df.columns if "MagpieData mean" in c]
    props = [c.replace("MagpieData mean ", "") for c in mean_cols]
    
    # Création du DataFrame des propriétés atomiques
    atom_props = pd.DataFrame(index=elements, columns=props)
    X_comp = df[elements].fillna(0)
    
    # Régression linéaire pour retrouver les propriétés atomiques brutes
    for p in props:
        y_p = df[f"MagpieData mean {p}"]
        mask = y_p.notna()
        if mask.sum() > 5: # Sécurité
            lr = LinearRegression(fit_intercept=False).fit(X_comp.loc[mask], y_p.loc[mask])
            atom_props[p] = lr.coef_
        else:
            atom_props[p] = 0.0

    return atom_props, props

def calculate_magpie_lite_single(comp_dict, atom_props_df, props_list, feature_columns_order):
    """
    Calcule TOUTES les statistiques Magpie pour correspondre aux attentes du modèle.
    Gère : mean, minimum, maximum, range, mode, avg_dev
    """
    # 1. Création du vecteur de composition
    x_vec = np.zeros(len(atom_props_df))
    for el, frac in comp_dict.items():
        if el in atom_props_df.index:
            idx = atom_props_df.index.get_loc(el)
            x_vec[idx] = frac
            
    feats = {}
    
    # 2. Boucle sur chaque propriété physique (Electronegativity, Radius, etc.)
    for prop in props_list:
        p_vec = atom_props_df[prop].values
        
        # MEAN
        mean_val = np.dot(x_vec, p_vec)
        feats[f"MagpieData mean {prop}"] = mean_val
        
        # Préparation pour les autres stats (valeurs des éléments présents)
        mask = x_vec > 0
        if not any(mask):
            p_present = np.array([0.0])
            fractions = np.array([1.0])
        else:
            p_present = p_vec[mask]
            fractions = x_vec[mask]

        # MIN / MAX / RANGE
        feats[f"MagpieData minimum {prop}"] = np.min(p_present)
        feats[f"MagpieData maximum {prop}"] = np.max(p_present)
        feats[f"MagpieData range {prop}"] = np.max(p_present) - np.min(p_present)
        
        # MODE (Propriété de l'élément le plus abondant)
        idx_mode = np.argmax(x_vec) # Index de l'élément majoritaire
        feats[f"MagpieData mode {prop}"] = p_vec[idx_mode]
        
        # AVG_DEV (Déviation moyenne absolue pondérée)
        # Formule : sum( x_i * |p_i - mean| )
        feats[f"MagpieData avg_dev {prop}"] = np.sum(x_vec * np.abs(p_vec - mean_val))

    # 3. Alignement avec les features attendues par le modèle
    df_res = pd.DataFrame([feats])
    
    # On crée un DataFrame final vide avec exactement les colonnes attendues
    df_final = pd.DataFrame(columns=feature_columns_order)
    
    # On remplit avec les valeurs calculées, 0 sinon
    for col in feature_columns_order:
        if col in df_res.columns:
            df_final.loc[0, col] = df_res.iloc[0][col]
        else:
            df_final.loc[0, col] = 0.0 # Feature manquante (ex: issue d'une autre prop non calculée)
            
    return df_final

# ==============================================================================
# 3. CHARGEMENT DES RESSOURCES
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_data = os.path.join(current_dir, 'data', 'training_data.csv')
        path_model_onset = os.path.join(current_dir, 'models', 'model_onset.joblib')
        path_model_tafel = os.path.join(current_dir, 'models', 'model_tafel.joblib')

        # Chargement
        if not os.path.exists(path_data): return None, None, None, None, None, None
        df_train = pd.read_csv(path_data)
        
        model_onset = joblib.load(path_model_onset)
        model_tafel = joblib.load(path_model_tafel)
        
        # Calibration Physique
        atom_props, all_props = learn_and_patch_physics(df_train, ALL_ELEMS)
        
        # --- RÉCUPÉRATION DES NOMS DE FEATURES ATTENDUS ---
        # On regarde dans le modèle ce qu'il a vu lors du "fit"
        # L'objet s'appelle souvent feature_names_in_ ou est accessible via le premier step
        try:
            # Pour un Pipeline, on regarde souvent le premier step (ex: 'imputer')
            if hasattr(model_onset, 'feature_names_in_'):
                feature_cols = model_onset.feature_names_in_
            elif hasattr(model_onset, 'regressor_'): # Si TransformedTargetRegressor
                 # On essaie d'accéder au pipeline interne
                 if hasattr(model_onset.regressor_, 'feature_names_in_'):
                     feature_cols = model_onset.regressor_.feature_names_in_
                 else:
                     # Fallback : on prend toutes les colonnes Magpie du CSV d'entrainement
                     feature_cols = [c for c in df_train.columns if c.startswith("MagpieData")]
        except:
             feature_cols = [c for c in df_train.columns if c.startswith("MagpieData")]

        return model_onset, model_tafel, atom_props, all_props, list(feature_cols), df_train
        
    except Exception as e:
        st.error(f"Erreur chargement : {str(e)}")
        return None, None, None, None, None, None

model_onset, model_tafel, atom_props, all_props, feature_cols, df_train = load_assets()

# ==============================================================================
# 4. INTERFACE UTILISATEUR
# ==============================================================================
st.title("⚗️ High-Performance HER Catalyst Predictor")
st.markdown("""
**Conception générative d'électrocatalyseurs** assistée par l'IA.
Entrez une formule chimique pour prédire son **Potentiel d'Onset** et sa **Pente de Tafel**.
""")

with st.sidebar:
    st.header("Paramètres")
    st.info("Modèle calibré sur 180 alliages expérimentaux.")
    st.markdown("### Éléments Supportés")
    st.write(", ".join(ALL_ELEMS))
    st.markdown("---")
    st.caption("Développé avec 🧠 GPR & ⚛️ Magpie Lite")

col1, col2 = st.columns([2, 1])
with col1:
    formula_input = st.text_input("Formule Chimique (ex: Pt90Ni10, Co30Fe20Ni50)", "Pt50Ni50")

if st.button("🚀 Prédire la Performance", type="primary"):
    if model_onset is None:
        st.error("Ressources non chargées. Vérifiez les fichiers.")
    else:
        comp_dict, error = parse_formula(formula_input)
        
        if error:
            st.warning(error)
        else:
            # Calcul des features alignées
            X_input = calculate_magpie_lite_single(comp_dict, atom_props, all_props, feature_cols)
            
            try:
                # Prédiction
                pred_onset = model_onset.predict(X_input)[0]
                pred_tafel = model_tafel.predict(X_input)[0]

                st.success(f"Composition analysée : {comp_dict}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("⚡ Onset Potential", f"{pred_onset:.1f} mV", delta="Plus bas est mieux", delta_color="inverse")
                with res_col2:
                    st.metric("📉 Tafel Slope", f"{pred_tafel:.1f} mV/dec", delta="Plus bas est mieux", delta_color="inverse")
                
                score = (pred_onset + pred_tafel) / 2
                if score < 40: st.balloons(); st.info("🌟 Candidat Exceptionnel !")
                elif score < 60: st.info("✅ Bon candidat.")
                else: st.warning("⚠️ Performance modeste.")
                    
            except Exception as e:
                st.error(f"Erreur prédiction : {e}")
