---
title: Apartment Price Prediction Zurich
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.9.0"
python_version: "3.12"
app_file: app.py
pinned: false
---

# Model Iterations Documentation

## Task: Apartment Price Prediction (Regression)

---

## Project Overview
This project predicts apartment rental prices in the canton of Zurich using machine learning.

The final solution includes:
- a trained regression model
- newly engineered features
- a working web application
- a documented iterative modeling process
- a completed README for Hugging Face deployment

The goal was to build a simple but improved end-to-end application that predicts monthly apartment rent based on apartment characteristics, location, and selected feature engineering variables.

---

## Summary of Iterative Process

| Iteration | Objective | Key Changes | Models Used | CV Mean R² | CV Std Dev | Change in Performance | Fit Diagnosis |
|------------|------------|-------------|-------------|------------|------------|-----------------------|----------------|
| **1** | Build a clean baseline model | - Used `apartments_data_enriched_lat_lon_combined.csv` as main dataset  <br> - Selected only relevant features  <br> - Added preprocessing pipeline  <br> - Applied 5-fold cross-validation | Linear Regression  <br> Random Forest (`n_estimators=150`, `random_state=42`) | **0.6492 (RF)**  <br> **-0.4294 (LR)** | **0.0470 (RF)**  <br> **0.7600 (LR)** | Baseline | ☑ Mixed baseline fit |
| **2** | Test stronger models and improve feature representation | - Added engineered location feature `distance_to_center`  <br> - Added text-based binary features from `description_raw`  <br> - Tuned Random Forest  <br> - Tested Gradient Boosting | Tuned Random Forest (`n_estimators=300`, `max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`)  <br> Gradient Boosting (`n_estimators=300`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`) | **0.6409 (RF Tuned)**  <br> **0.6116 (GBR)** | **0.0456 (RF Tuned)**  <br> **0.0260 (GBR)** | No improvement over baseline Random Forest | ☑ Good comparison, baseline remained best |

---

## Dataset
The main dataset used for the final model is:

`apartments_data_enriched_lat_lon_combined.csv`

This dataset already contains enriched apartment, municipality, and location data, including:
- apartment characteristics
- postal code and town
- geographic coordinates
- municipality statistics
- tax-related indicators

### Target Variable
- `price`

---

## Preprocessing
The following preprocessing steps were applied:

- Loaded the enriched apartment dataset
- Defined `price` as the target variable
- Selected only relevant model features
- Cleaned the `town` and `description_raw` columns
- Automatically separated numerical and categorical variables
- Imputed missing numerical values using the median
- Imputed missing categorical values using the most frequent value
- Scaled numerical variables using `StandardScaler`
- Encoded categorical variables using `OneHotEncoder`
- Combined preprocessing with a `ColumnTransformer`
- Embedded preprocessing and model training inside a scikit-learn pipeline
- Evaluated all models with 5-fold cross-validation

---

## Feature Engineering

### Main Engineered Feature
- **`distance_to_center`**

This feature represents the distance between an apartment and the center of Zurich, calculated from latitude and longitude using the Haversine formula.

### Additional Engineered Features
The following binary features were created from `description_raw`:

- `is_luxury`
- `is_temporary`
- `is_furnished`
- `is_attika`
- `is_loft`

### Why these features?
These features capture important apartment characteristics that are relevant for rental prices:

- apartments closer to Zurich city center are often more expensive
- luxury apartments tend to have higher rents
- furnished and temporary apartments often follow different pricing patterns
- attica and loft apartments may reflect premium property types

---

## Final Selected Features
The final model was trained using the following feature set:

- `rooms`
- `area`
- `postalcode`
- `town`
- `pop`
- `pop_dens`
- `frg_pct`
- `emp`
- `tax_income`
- `distance_to_center`
- `is_luxury`
- `is_temporary`
- `is_furnished`
- `is_attika`
- `is_loft`

---

## Models Used
The following regression models were tested:

- Linear Regression
- Random Forest Regressor
- Tuned Random Forest Regressor
- Gradient Boosting Regressor

---

## Evaluation Method
The models were evaluated using **5-fold cross-validation**.

The following regression metrics were used:

- **R²**
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

---

## Final Selected Model
The final selected model is:

**Random Forest Regressor**

### Final Model Performance
- **R²:** 0.6492
- **CV Std Dev (R²):** 0.0470
- **MAE:** 428.66
- **RMSE:** 647.14

### Why this model was selected
Random Forest Regressor achieved the best overall performance among all tested models in cross-validation.

It produced:
- the highest mean R²
- the lowest MAE
- the lowest RMSE

Although stronger and more complex models were tested in iteration 2, the baseline Random Forest remained the best-performing and most reliable model for this feature set.

---

## Application
The application allows users to:

- enter the number of rooms
- enter the apartment size in square meters
- select a postal code in the canton of Zurich
- select a town with automatic synchronization to postal code
- choose apartment features such as furnished, temporary, luxury, attika, and loft
- receive an instant monthly rent prediction
- see the engineered feature `distance_to_center` directly in the interface

### Application Link
**https://huggingface.co/spaces/fetaiedi/apartment-price-prediction-zurich**

---

## Project Files
- `train.py` – model training, evaluation, and model selection
- `app.py` – Gradio web application
- `best_model.pkl` – trained final model
- `requirements.txt` – required Python packages
- `README.md` – project documentation

---

## Author
**Edin Fetai**