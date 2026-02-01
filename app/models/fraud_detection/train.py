import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier


# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv("data/creditcard.csv")

print("Total rows:", len(df))
print("Fraud cases:", df["Class"].sum())
print("Fraud rate:", df["Class"].mean())


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================

# Scale Amount (V1-V28 already PCA scaled)
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
df = df.drop(columns=["Amount"])

# Optional: keep Time or drop
df = df.drop(columns=["Time"])  # often removed


# =====================================================
# 3. FEATURES & TARGET
# =====================================================
X = df.drop(columns=["Class"])
y = df["Class"]

feature_columns = X.columns.tolist()


# =====================================================
# 4. TRAIN / TEST SPLIT (STRATIFIED)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train fraud rate:", y_train.mean())
print("Test fraud rate:", y_test.mean())


# =====================================================
# 5. HANDLE CLASS IMBALANCE
# =====================================================
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
print("Scale_pos_weight:", scale_pos_weight)


# =====================================================
# 6. TRAIN XGBOOST
# =====================================================
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)


# =====================================================
# 7. PROBABILITY PREDICTIONS
# =====================================================
y_proba = model.predict_proba(X_test)[:, 1]


# =====================================================
# 8. CORE METRICS
# =====================================================
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("\n===== MODEL PERFORMANCE =====")
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


# =====================================================
# 9. THRESHOLD OPTIMIZATION (F1)
# =====================================================
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

precision = precision[:-1]
recall = recall[:-1]

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_index = np.argmax(f1_scores)

chosen_threshold = thresholds[best_index]

print("\nBest threshold (F1 optimized):", round(chosen_threshold, 4))
print("Precision:", round(precision[best_index], 4))
print("Recall:", round(recall[best_index], 4))


# =====================================================
# 10. FINAL PREDICTIONS
# =====================================================
y_pred = (y_proba >= chosen_threshold).astype(int)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# =====================================================
# 11. COST ANALYSIS (BUSINESS VIEW)
# =====================================================
avg_fraud_amount = df[df["Class"] == 1]["Amount_scaled"].mean()

true_frauds_caught = cm[1][1]
false_negatives = cm[1][0]

print("\n===== BUSINESS IMPACT =====")
print("Frauds caught:", true_frauds_caught)
print("Frauds missed:", false_negatives)


# =====================================================
# 12. SAVE EVERYTHING
# =====================================================
os.makedirs("models_fraud", exist_ok=True)

joblib.dump(model, "models_fraud/fraud_model_v1.pkl")
joblib.dump(chosen_threshold, "models_fraud/fraud_threshold.pkl")
joblib.dump(feature_columns, "models_fraud/fraud_feature_columns.pkl")
joblib.dump(scaler, "models_fraud/fraud_amount_scaler.pkl")

print("\nModel + threshold + schema saved successfully.")
