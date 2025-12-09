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
* `data/training_data.csv`: Experimental data used to calibrate the physics engine.
* `models/model_*.joblib`: Pre-trained AI models.

## 🔬 Methodology
The AI relies on a **Physics-Informed** approach. Rather than using purely statistical descriptors, the model has learned the relationships between fundamental atomic properties (electronegativity, atomic radius, etc.) and catalytic activity.

---
*Developed for materials science research.*
