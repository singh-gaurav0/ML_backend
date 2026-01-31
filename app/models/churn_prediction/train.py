import pandas as pd
import joblib
import logging
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import os
import shap

# ---------------------------
# Setup Logging
# ---------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Time-based churn training started")

# ---------------------------
# Load Data
# ---------------------------
accounts = pd.read_csv("data/ravenstack_accounts.csv")
subscriptions = pd.read_csv("data/ravenstack_subscriptions.csv")
tickets = pd.read_csv("data/ravenstack_support_tickets.csv")
churn_events = pd.read_csv("data/ravenstack_churn_events.csv")

# ---------------------------
# Convert to datetime
# ---------------------------
accounts["signup_date"] = pd.to_datetime(accounts["signup_date"])
subscriptions["start_date"] = pd.to_datetime(subscriptions["start_date"])
subscriptions["end_date"] = pd.to_datetime(subscriptions["end_date"], errors="coerce")
tickets["submitted_at"] = pd.to_datetime(tickets["submitted_at"])
churn_events["churn_date"] = pd.to_datetime(churn_events["churn_date"])

# ---------------------------
# Define Cutoff
# ---------------------------
cutoff_date = pd.Timestamp("2024-09-01")
prediction_window = 60

# ---------------------------
# Create Time-Based Label
# ---------------------------
churn_events["churn_within_window"] = (
    (churn_events["churn_date"] > cutoff_date) &
    (churn_events["churn_date"] <= cutoff_date + pd.Timedelta(days=prediction_window))
)

churn_accounts = churn_events[churn_events["churn_within_window"]]["account_id"]
accounts["churn_label"] = accounts["account_id"].isin(churn_accounts).astype(int)

print("Churn rate:", accounts["churn_label"].mean())

# ---------------------------
# Filter Before Cutoff
# ---------------------------
subscriptions_before = subscriptions[subscriptions["start_date"] <= cutoff_date]
tickets_before = tickets[tickets["submitted_at"] <= cutoff_date]

# ---------------------------
# Active Subscription Feature
# ---------------------------
subscriptions["active_at_cutoff"] = (
    (subscriptions["start_date"] <= cutoff_date) &
    (
        (subscriptions["end_date"].isna()) |
        (subscriptions["end_date"] > cutoff_date)
    )
)

active_agg = subscriptions.groupby("account_id").agg(
    active_subscription_at_cutoff=("active_at_cutoff", "max")
).reset_index()

# ---------------------------
# Subscription Aggregates
# ---------------------------
sub_agg = subscriptions_before.groupby("account_id").agg(
    total_mrr=("mrr_amount", "sum"),
    avg_seats=("seats", "mean"),
    upgrade_count=("upgrade_flag", "sum"),
    downgrade_count=("downgrade_flag", "sum"),
    auto_renew_ratio=("auto_renew_flag", "mean"),
    subscription_count=("subscription_id", "count"),
    last_subscription_date=("start_date", "max")
).reset_index()

sub_agg["days_since_last_subscription"] = (
    cutoff_date - sub_agg["last_subscription_date"]
).dt.days

sub_agg.drop(columns=["last_subscription_date"], inplace=True)

# ---------------------------
# Ticket Aggregates
# ---------------------------
ticket_agg = tickets_before.groupby("account_id").agg(
    ticket_count=("ticket_id", "count"),
    avg_resolution_time=("resolution_time_hours", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    escalation_ratio=("escalation_flag", "mean"),
    last_ticket_date=("submitted_at", "max")
).reset_index()

ticket_agg["days_since_last_ticket"] = (
    cutoff_date - ticket_agg["last_ticket_date"]
).dt.days

ticket_agg.drop(columns=["last_ticket_date"], inplace=True)

# ---------------------------
# Merge
# ---------------------------
df = accounts.merge(sub_agg, on="account_id", how="left")
df = df.merge(ticket_agg, on="account_id", how="left")
df = df.merge(active_agg, on="account_id", how="left")

df.fillna(0, inplace=True)
df = df[df["signup_date"] <= cutoff_date]

df["tenure_days"] = (cutoff_date - df["signup_date"]).dt.days

# ---------------------------
# 🔥 Behavioral Feature Engineering
# ---------------------------
df["downgrade_ratio"] = df["downgrade_count"] / df["subscription_count"].replace(0, 1)
df["support_intensity"] = df["ticket_count"] / df["tenure_days"].replace(0, 1)
df["support_pain"] = df["avg_resolution_time"] * df["escalation_ratio"]
df["engagement_gap"] = df["days_since_last_ticket"] / df["tenure_days"].replace(0, 1)

# ---------------------------
# Feature Selection
# ---------------------------
feature_columns = [
    "total_mrr",
    "avg_seats",
    "downgrade_ratio",
    "auto_renew_ratio",
    "days_since_last_subscription",
    "support_intensity",
    "support_pain",
    "engagement_gap",
    "active_subscription_at_cutoff",
    "tenure_days"
]

X = df[feature_columns]
y = df["churn_label"]

# Save feature schema
os.makedirs("models_churn/churn_prediction", exist_ok=True)
joblib.dump(feature_columns, "models_churn/churn_prediction/churn_feature_columns.pkl")

# ---------------------------
# Time-Based Split
# ---------------------------
df_sorted = df.sort_values("signup_date").reset_index(drop=True)
split_index = int(len(df_sorted) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ---------------------------
# Train Model (Controlled Capacity)
# ---------------------------
scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

model = XGBClassifier(
    n_estimators=60,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=10,
    reg_alpha=5,
    min_child_weight=10,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Train AUC:", train_auc)
print("Test AUC:", test_auc)

# ---------------------------
# Save Model
# ---------------------------
model_path = "models_churn/churn_prediction/churn_model_time_v6.pkl"
joblib.dump(model, model_path)

print("Model saved successfully.")
