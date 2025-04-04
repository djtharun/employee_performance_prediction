# Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import joblib
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load Dataset
df = pd.read_csv("HR-Employee-Attrition.csv") 

# Step 2: Initial EDA
print("First 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Step 3: Handle Missing Values (Simple Imputation)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(method='ffill', inplace=True)  # For categorical if needed

# Step 4: Encode Categorical Variables
categorical_cols = df.select_dtypes(include='object').columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Feature Scaling
scaler = StandardScaler()
scaled_cols = ['Salary', 'Work_Hours', 'Age', 'YearsAtCompany']
for col in scaled_cols:
    if col in df.columns:
        df[col] = scaler.fit_transform(df[[col]])

# Step 6: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Step 7: Regression Task — Predict Performance Score
if 'PerformanceRating' in df.columns:
    features_reg = df.drop(['PerformanceRating', 'Attrition'], axis=1, errors='ignore')
    target_reg = df['PerformanceRating']

    X_train, X_test, y_train, y_test = train_test_split(features_reg, target_reg, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print("\n--- Regression Results (Performance Prediction) ---")
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² Score:", r2_score(y_test, y_pred))

    joblib.dump(regressor, "performance_model.pkl")

# Step 8: Classification Task — Predict Retention
if 'Attrition' in df.columns:
    features_cls = df.drop(['PerformanceRating', 'Attrition'], axis=1, errors='ignore')
    target_cls = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(features_cls, target_cls, test_size=0.2, random_state=42)

    # Compare Classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n--- {name} ---")
        print("Accuracy:", acc)
        print("Classification Report:\n", classification_report(y_test, preds))
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("ROC AUC:", roc_auc_score(y_test, preds))

    # Save best model
    best_model = classifiers["XGBoost"]
    joblib.dump(best_model, "retention_model.pkl")

# Step 9: Done! You now have:
# ✔ Cleaned dataset
# ✔ Two trained models (regression + classification)
# ✔ Heatmap and EDA
# ✔ Model saved with joblib
# ✔ Ready for Streamlit/Flask deployment
