# ⚗️ HER Catalyst Predictor

**An AI-powered web application for the accelerated discovery of new electrocatalysts (Hydrogen Evolution Reaction).**

This application uses Gaussian Process Regression (GPR) trained on an experimental database ("Small Data Regime") to instantly predict the performance of new complex alloys (Ternary, Quaternary, Quinary).

## 🚀 Features
* **Instant Prediction:** Enter a chemical formula (e.g., `Co30Fe20Ni50`) and get its estimated performance.
* **"Magpie Lite" Engine:** Integrates an ultra-lightweight physicochemical descriptor calculation engine, requiring no heavy external database.
* **Dual Target:** Simultaneous prediction of *Onset Potential* (mV) and *Tafel Slope* (mV/dec).

## 📂 Project Structure
* `app.py`: The core of the Streamlit application (Interface & Calculation Engine).
* `requirements.txt`: List of required Python libraries.
* `data/her_catalysts_dataset_v1.csv`: Experimental data used to train the AI models.
* `data/dataset_metadata.csv`: A structured metadata dictionary defining the HER Catalysts dataset variables. It maps identifiers, experimental targets (Onset Potential, Tafel Slope), and physicochemical descriptors (Magpie features) to their scientific definitions to ensure documentation and reproducibility.
* `models/model_*.joblib`: Pre-trained AI models.

## 🔬 Methodology
The AI relies on a **Physics-Informed** approach. Rather than using purely statistical descriptors, the model has learned the relationships between fundamental atomic properties (electronegativity, atomic radius, etc.) and catalytic activity.

---
*Developed for materials science research.*
