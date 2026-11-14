
# Week 4: Predictive Modeling in Logistics Systems
# Educational project using simulated logistics data.

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("hypothetical_logistics_dataset_week4.csv")

X = df[["Distance_km", "Weight_kg", "Traffic", "Weather"]]
y = df["Delivery_Time_Days"]

numeric_features = ["Distance_km", "Weight_kg"]
categorical_features = ["Traffic", "Weather"]

def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = {}
pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results[name] = {
        "MAE (days)": mean_absolute_error(y_test, predictions),
        "RMSE (days)": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2 Score": r2_score(y_test, predictions)
    }

    pipelines[name] = pipeline
    print("\n", name, results[name])

# Five-fold cross-validation
rf_pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", RandomForestRegressor(
        n_estimators=100, random_state=42
    ))
])

scores = cross_val_score(
    rf_pipeline, X, y, cv=5,
    scoring="neg_mean_absolute_error"
)

print("5-fold CV MAE:", -scores.mean())

# Select model with lowest test MAE
comparison = pd.DataFrame(results).T
comparison.to_csv("week4_model_comparison.csv")

best_name = comparison["MAE (days)"].idxmin()
best_model = pipelines[best_name]

print("Selected model:", best_name)

# Example prediction
sample = pd.DataFrame({
    "Distance_km": [300],
    "Weight_kg": [100],
    "Traffic": ["Medium"],
    "Weather": ["Clear"]
})

print("Predicted delivery days:",
      best_model.predict(sample)[0])
