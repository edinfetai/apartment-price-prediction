---
title: Apartment Price Prediction Zurich
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Model Iterations Documentation

## Task: Apartment Price Prediction (Regression)

---

## Project Overview
This project predicts apartment rental prices in the canton of Zurich using machine learning.

The solution includes:
- a trained regression model
- one newly engineered feature
- a working web application
- a documented iterative modeling process
- a completed README file for Hugging Face deployment

---

## Summary of Iterative Process

| Iteration | Objective | Key Changes | Models Used | CV Mean R² | CV Std Dev | Change in Performance | Fit Diagnosis |
|------------|------------|-------------|-------------|------------|------------|-----------------------|----------------|
| **1** | Build baseline model | - Loaded enriched apartment dataset  <br> - Missing value imputation  <br> - One-hot encoding  <br> - Standard scaling  <br> - 5-fold cross-validation | Linear Regression  <br> Random Forest (`n_estimators=100`, `random_state=42`) | **0.9088 (RF)**  <br> **-11.0937 (LR)** | **0.0594 (RF)**  <br> **11.7237 (LR)** | Baseline | ☑ Overfitting ☐ Underfitting ☐ Good Fit |
| **2** | Improve predictive performance | - Added new feature `distance_to_center`  <br> - Recomputed engineered variables  <br> - Compared stronger models  <br> - Hyperparameter tuning  <br> - 5-fold cross-validation | Tuned Random Forest (`n_estimators=300`, `max_depth=15`, `min_samples_split=5`, `random_state=42`)  <br> Gradient Boosting (`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`) | **0.9453 (GBR)**  <br> **0.9053 (RF Tuned)** | **0.0312 (GBR)**  <br> **0.0604 (RF Tuned)** | **+0.0365 improvement** over best baseline model | ☐ Overfitting ☐ Underfitting ☑ Good Fit |

---

## Preprocessing
The following preprocessing steps were applied:
- Loaded the dataset `apartments_data_enriched_with_new_features.csv`
- Defined `price` as the target variable
- Split the data into input features (`X`) and target (`y`)
- Automatically separated numerical and categorical variables
- Imputed missing numerical values using the median
- Imputed missing categorical values using the most frequent value
- Scaled numerical features with `StandardScaler`
- Encoded categorical variables with `OneHotEncoder`
- Combined preprocessing steps using a `ColumnTransformer`
- Embedded preprocessing and model training in a scikit-learn pipeline
- Evaluated all models using 5-fold cross-validation

---

## Notes

**Metric:**  
R², MAE and RMSE were used for model comparison with 5-fold cross-validation.

**Created Feature:**  
- `distance_to_center`

**Reason for Feature Selection:**  
The distance to Zurich city center is relevant because apartments closer to the center are generally more expensive.  
This feature was calculated using latitude and longitude with the Haversine formula.

**Final Selected Features:**  
- rooms
- area
- postalcode
- town
- lat
- lon
- furnished
- temporary
- luxurious
- zurich_city
- room_per_m2
- distance_to_center  
plus the additional variables contained in the enriched dataset after preprocessing and encoding.

**Reason for Model Selection:**  
Gradient Boosting Regressor was selected because it achieved the best overall cross-validation performance:
- **R²:** 0.9453
- **MAE:** 158.31
- **RMSE:** 238.43

It outperformed all other tested models and also showed the lowest R² standard deviation among the best-performing models, indicating strong and stable generalization.

---

## Final Selected Model
**Gradient Boosting Regressor**

### Final Performance
- **R²:** 0.9453
- **MAE:** 158.31
- **RMSE:** 238.43
- **CV Std Dev (R²):** 0.0312

---

## Application
The application allows users to:
- enter the number of rooms
- enter the apartment area in square meters
- select a postal code in the canton of Zurich
- select a town with automatic synchronization to postal code
- choose apartment features
- receive an instant prediction of the monthly rental price

### Application Link
**[Insert Hugging Face Space link here]**

---

## Project Files
- `train.py` – model training, evaluation, and selection
- `app.py` – Gradio web application
- `best_model.pkl` – trained final model
- `requirements.txt` – required Python packages
- `README.md` – project documentation

---

## Author
**Edin Fetai**
