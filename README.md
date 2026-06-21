# Heart Disease Risk Prediction — ML Dashboard & App

An end-to-end data science capstone that predicts heart disease risk from a large
public-health survey and explains the key risk factors through an interactive app
and a Tableau dashboard.

> **Disclaimer:** This is an educational data-science project. It is **not** intended
> for clinical diagnosis, treatment decisions, or individual medical advice.

---

## Project Overview

This project uses the CDC BRFSS 2015 health survey (**253,680 records, 22 columns**)
to predict the target `HeartDiseaseorAttack`. Only **9.4%** of records are positive,
so the dataset is imbalanced — which shaped the whole modelling approach.

The workflow runs from data exploration through feature engineering, model
comparison, explainability, and two user-facing deliverables: a Gradio app and a
Tableau dashboard.

## Dataset

- **Source:** CDC BRFSS 2015 (Kaggle: `alexteboul/heart-disease-health-indicators-dataset`)
- **Size:** 253,680 rows × 22 columns (21 features + binary target)
- **Target:** `HeartDiseaseorAttack` — 9.4% positive (imbalanced)
- **Note:** Self-reported survey data; no clinical measurements. BRFSS is run annually,
  so fresh data is available each year.

## Repository Structure

| Folder / File | Contents |
|---|---|
| `Notebooks/` | The 5 project notebooks (loading & EDA, training, SHAP, app prep, Tableau prep) |
| `app/` | `app.py` — the Gradio prediction app |
| `models/` | Trained model, scaler (`.pkl`), and result summary CSVs |
| `dashboard_graphs/` | Interactive Plotly charts (HTML) |
| `tableau_dashboard_data/` | Summary CSVs prepared for Tableau |
| `Heart Disease Dashboard.twbx` | Packaged Tableau dashboard |
| `Heart_Disease_Capstone_Slides.pptx` | Presentation slides |
| `requirements.txt` | Python dependencies |

## Method

1. **EDA** — examined the target imbalance and heart-disease rate by risk factor.
2. **Feature engineering** — one-hot encoded `Diabetes` (categories, not a scale),
   engineered `risk_factor_count` and `total_poor_health_days`; 22 raw columns → 24
   model-ready features.
3. **Modelling** — compared **Logistic Regression, Random Forest, and XGBoost** with a
   stratified split, scaling fit on training data only, and `class_weight=balanced`.
4. **Explainability** — used SHAP to show which features drive predictions.

## Results

| Model | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.75 | **0.80** | 0.25 | 0.38 | 0.85 |
| Random Forest | 0.90 | 0.10 | 0.37 | 0.16 | 0.81 |
| XGBoost | 0.91 | 0.09 | 0.55 | 0.16 | 0.85 |

**Chosen model: Logistic Regression.** On imbalanced data, accuracy is misleading —
Random Forest and XGBoost score ~90% accuracy but catch only ~10% of real cases.
Logistic Regression catches ~80% (**recall 0.80**) with the best ROC-AUC, and it's
interpretable. For risk screening, a missed case is far worse than a false alarm, so
recall was prioritised over precision.

SHAP confirmed the model relies on clinically sensible factors (age, general health,
cholesterol, blood pressure), and the engineered `risk_factor_count` ranked in the
top 5 — evidence that the feature engineering helped.

## How to Run

```bash
pip install -r requirements.txt

# Run the app
cd app
python app.py
# opens at http://127.0.0.1:7860
```

The notebooks in `Notebooks/` can be run in order (01 → 05) to reproduce the full
pipeline from the Kaggle dataset.

## Tools & Libraries

Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn ·
Plotly · Gradio · Tableau · joblib · Jupyter

## Author

**Amna Ghaffar** — Data Science Capstone
