# Automobile Insurance Fraud Detection

An end-to-end machine learning system for detecting potentially fraudulent **automobile insurance claims**—from data preparation and model training to an interactive **Streamlit** web app for real-time risk scoring.

## Why this matters

Insurance fraud is a real, costly problem: it increases premiums for honest customers and creates operational burden for claims teams. In practice, fraud signals are often subtle and the data is typically **imbalanced** (many more normal claims than fraud). This project focuses on **high-recall detection** and **PR-AUC** (precision–recall performance), which better reflects real-world outcomes than accuracy.

## What this project delivers

- **Fraud risk scoring UI**: Streamlit app that outputs a fraud probability and lets you tune the decision threshold.
- **Leakage-safe pipeline**: preprocessing + scaling + SMOTE done correctly (fit only on training folds/sets).
- **Production-friendly artifacts**: saved model (and scaler) stored under `models/`.
- **Reproducible notebooks**: EDA → modeling workflow in `notebooks/`.

## Key functionality

- Data cleaning + feature engineering (e.g., `days_to_incident`, age/tenure groups)
- Ordinal encoding for severity + one-hot encoding for categorical variables
- Model training with **XGBoost** (optimized for imbalanced classification)
- Optional oversampling with **SMOTE** (applied only to training data)
- Evaluation with metrics that matter for fraud: **Recall** and **PR-AUC**

## Tech stack

- Python, pandas, NumPy, scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- Streamlit

## Installation (GitHub)

Clone and set up a virtual environment from the project root:

```bash
git clone <your-repo-url>
cd Insurance_Project

python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the web app

```bash
./venv/bin/streamlit run app.py
```

The app:
- collects claim attributes in the sidebar,
- applies the same preprocessing used during training,
- outputs fraud probability + prediction at your chosen threshold.

## Reproduce training (notebooks)

Start Jupyter **from the project root**:

```bash
source venv/bin/activate
jupyter lab
```

Recommended order:
1. `notebooks/00_start_here.ipynb`
2. `notebooks/01_eda.ipynb`
3. `notebooks/02_modeling.ipynb`

## Project structure

- `data/` – raw + cleaned datasets
- `notebooks/` – analysis + training workflow
- `src/insurance_project/` – reusable preprocessing/modeling code
- `models/` – saved artifacts (e.g., `best_xgboost_v1.pkl`, scaler)
- `app.py` – Streamlit application

## (Optional) Rebuild the cleaned dataset

```bash
source venv/bin/activate
python main.py
```

## Notes for real-world use

- This project is for educational/demo purposes; real claims systems require strong privacy controls, monitoring, and human-in-the-loop review.
- Always validate models on unseen data, monitor drift, and periodically retrain as fraud patterns evolve.
