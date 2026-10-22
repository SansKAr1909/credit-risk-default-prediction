import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

target_col = 'SeriousDlqin2yrs'

# ============================================================
# LOAD DATA
# ============================================================
train_clean = pd.read_csv('data_train_clean.csv')
val_clean = pd.read_csv('data_val_clean.csv')
test_clean = pd.read_csv('data_test_clean.csv')

X_train = train_clean.drop(columns=[target_col])
y_train = train_clean[target_col]
X_val = val_clean.drop(columns=[target_col])
y_val = val_clean[target_col]
X_test = test_clean.drop(columns=[target_col])
y_test = test_clean[target_col]

# ============================================================
# MODEL COMPARISON
# ============================================================

# --- Decision Tree (unrestricted) 
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Unrestricted Tree - Train AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
print("Unrestricted Tree - Val AUC:", roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

# --- Decision Tree (restricted) ---
model_restricted = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=42)
model_restricted.fit(X_train, y_train)
print("Restricted Tree - Train AUC:", roc_auc_score(y_train, model_restricted.predict_proba(X_train)[:, 1]))
print("Restricted Tree - Val AUC:", roc_auc_score(y_val, model_restricted.predict_proba(X_val)[:, 1]))

# --- Random Forest ---
model_rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, random_state=42, n_jobs=-1)
model_rf.fit(X_train, y_train)
print("Random Forest - Train AUC:", roc_auc_score(y_train, model_rf.predict_proba(X_train)[:, 1]))
print("Random Forest - Val AUC:", roc_auc_score(y_val, model_rf.predict_proba(X_val)[:, 1]))

# --- XGBoost (final chosen model) ---
model_xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric='auc', random_state=42
)
model_xgb.fit(X_train, y_train)
print("XGBoost - Train AUC:", roc_auc_score(y_train, model_xgb.predict_proba(X_train)[:, 1]))
print("XGBoost - Val AUC:", roc_auc_score(y_val, model_xgb.predict_proba(X_val)[:, 1]))


# ============================================================
# FINAL EVALUATION ON TEST SET
# ============================================================
test_probs = model_xgb.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_probs)
print("\nFINAL - XGBoost Test ROC-AUC:", test_auc)

# ============================================================
# THRESHOLD EVALUATION (chosen threshold = 0.2)
# ============================================================
THRESHOLD = 0.2
test_preds = (test_probs >= THRESHOLD).astype(int)

print(f"\nConfusion Matrix at threshold={THRESHOLD}:")
print(confusion_matrix(y_test, test_preds))
print(classification_report(y_test, test_preds))

# ============================================================
# SAVE FINAL MODEL
# ============================================================
joblib.dump(model_xgb, 'final_xgb_model.pkl')
print("\nModel saved to final_xgb_model.pkl")