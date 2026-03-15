import pandas as pd
import numpy as np
import joblib

from math import radians, sin, cos, sqrt, atan2

from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ----------------------------
# 1. Load data
# ----------------------------
df = pd.read_csv("apartments_data_enriched_with_new_features.csv")


# ----------------------------
# 2. Feature engineering
# New feature: distance_to_center
# Zurich center coordinates (approx.)
# ----------------------------
ZURICH_LAT = 47.3769
ZURICH_LON = 8.5417

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return r * c

df["distance_to_center"] = df.apply(
    lambda row: haversine_distance(row["lat"], row["lon"], ZURICH_LAT, ZURICH_LON),
    axis=1
)


# ----------------------------
# 3. Define target and features
# ----------------------------
target = "price"

drop_cols = ["price"]

X = df.drop(columns=drop_cols)
y = df[target]


# ----------------------------
# 4. Detect numeric / categorical columns
# ----------------------------
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()


# ----------------------------
# 5. Preprocessing
# ----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ----------------------------
# 6. Iteration 1 models
# ----------------------------
iteration_1_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ----------------------------
# 7. Iteration 2 models
# ----------------------------
iteration_2_models = {
    "Random Forest Tuned": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ----------------------------
# 8. Evaluation helper
# ----------------------------
def evaluate_models(models, X, y, preprocessor, iteration_name):
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=5,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error"
            },
            n_jobs=-1
        )

        mean_r2 = scores["test_r2"].mean()
        mean_mae = -scores["test_mae"].mean()
        mean_rmse = -scores["test_rmse"].mean()

        results.append({
            "iteration": iteration_name,
            "model": model_name,
            "r2": mean_r2,
            "mae": mean_mae,
            "rmse": mean_rmse
        })

        print(f"{iteration_name} - {model_name}")
        print(f"  R2:   {mean_r2:.4f}")
        print(f"  MAE:  {mean_mae:.2f}")
        print(f"  RMSE: {mean_rmse:.2f}")
        print("-" * 40)

    return results


# ----------------------------
# 9. Run both iterations
# ----------------------------
all_results = []

all_results.extend(evaluate_models(iteration_1_models, X, y, preprocessor, "Iteration 1"))
all_results.extend(evaluate_models(iteration_2_models, X, y, preprocessor, "Iteration 2"))

results_df = pd.DataFrame(all_results)
print("\nSummary:")
print(results_df.sort_values(by="rmse"))


# ----------------------------
# 10. Select best model by RMSE
# ----------------------------
best_row = results_df.sort_values(by="rmse").iloc[0]
best_model_name = best_row["model"]

if best_model_name == "Linear Regression":
    best_model = LinearRegression()
elif best_model_name == "Random Forest":
    best_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
elif best_model_name == "Random Forest Tuned":
    best_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    )
else:
    best_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])

final_pipeline.fit(X, y)

joblib.dump(final_pipeline, "best_model.pkl")
print(f"\nBest model saved as best_model.pkl: {best_model_name}")
