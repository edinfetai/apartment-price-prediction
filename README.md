---
title: Apartment Price Prediction Zurich
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Apartment Price Prediction Application

## Project Overview
This project predicts apartment rental prices in the canton of Zurich using machine learning.  
It includes a trained regression model, a deployed web application, and a documented iterative modeling process.

The goal was to build a simple but functional end-to-end prediction system that allows users to enter apartment details and receive an estimated monthly rent.

---

## Objective
The objective of this project is to estimate the monthly rental price of apartments in the canton of Zurich based on selected apartment characteristics.

The solution includes:
- a trained regression model
- one newly engineered feature
- a working prediction app
- a documented iterative modeling process
- a completed README file for deployment and submission

---

## Dataset
The project uses the dataset:

`apartments_data_enriched_with_new_features.csv`

The dataset contains apartment-related information for the canton of Zurich.

### Target Variable
- `price`

### Example Input Features
- `rooms`
- `area`
- `postalcode`
- `town`
- `lat`
- `lon`
- `furnished`
- `temporary`
- `luxurious`
- `zurich_city`

---

## Preprocessing
The following preprocessing steps were applied:

- Loaded the apartment dataset from CSV
- Defined `price` as the target variable
- Split the data into input features (`X`) and target (`y`)
- Separated numerical and categorical variables automatically
- Imputed missing numerical values using the median
- Imputed missing categorical values using the most frequent value
- Scaled numerical variables using `StandardScaler`
- Encoded categorical variables using `OneHotEncoder`
- Combined preprocessing steps with a `ColumnTransformer`
- Embedded preprocessing and model training in a scikit-learn pipeline
- Evaluated the models using 5-fold cross-validation

---

## Feature Engineering
A new feature was created for this project:

### `distance_to_center`
This feature measures the distance between an apartment and the center of Zurich using latitude and longitude coordinates.

The feature was calculated using the **Haversine formula**, which estimates the distance between two geographic coordinates.

### Why this feature?
Distance to the city center is relevant because apartments closer to central Zurich are generally more expensive.  
This makes `distance_to_center` a meaningful predictor for rental prices.

---

## Iterative Modeling Process

### Iteration Summary Table

| Iteration | Objective | Changes Compared to Previous Iteration | Preprocessing Steps | Models Used | Hyperparameters | Cross-Validation Results | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Build a baseline model | Initial model setup with standard preprocessing | Missing value imputation, scaling of numeric features, one-hot encoding of categorical features, 5-fold cross-validation | Linear Regression, Random Forest Regressor | Linear Regression: default settings; Random Forest: `n_estimators=100`, `random_state=42` | Linear Regression: R² = -11.0937, MAE = 2315.36, RMSE = 2712.60; Random Forest: R² = 0.9088, MAE = 176.72, RMSE = 305.51 | Random Forest performed much better than Linear Regression and became the stronger baseline |
| 2 | Improve model performance | Added the new feature `distance_to_center` and compared stronger models | Same preprocessing as iteration 1 plus feature engineering for `distance_to_center` | Random Forest Regressor (Tuned), Gradient Boosting Regressor | Random Forest Tuned: `n_estimators=300`, `max_depth=15`, `min_samples_split=5`, `random_state=42`; Gradient Boosting: `n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `random_state=42` | Random Forest Tuned: R² = 0.9053, MAE = 182.71, RMSE = 312.70; Gradient Boosting: R² = 0.9453, MAE = 158.31, RMSE = 238.43 | Gradient Boosting achieved the best overall results and was selected as the final model |

---

## Models Used
The following regression models were tested during the project:

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

These metrics were used to compare the models across iterations and select the final model.

---

## Final Selected Model
The final selected model is:

**Gradient Boosting Regressor**

### Final Model Performance
- **R²:** 0.9453
- **MAE:** 158.31
- **RMSE:** 238.43

### Why this model was selected
Gradient Boosting Regressor achieved the best overall cross-validation performance among all tested models.  
It showed the highest R² and the lowest MAE and RMSE, which indicates the best balance between predictive accuracy and generalization.

---

## Application
The application allows users to enter apartment details and receive a predicted monthly rental price.

The app includes:
- manual input for rooms and area
- postal code selection
- town selection with automatic synchronization
- automatic background handling of latitude and longitude
- apartment feature selection
- instant rent prediction output

### Application Link
**[Insert Hugging Face Space link here]**

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
