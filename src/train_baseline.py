import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

target_col = 'SeriousDlqin2yrs'

# Load the already-saved scaled data
train_scaled = pd.read_csv('data_train_scaled.csv')
val_scaled = pd.read_csv('data_val_scaled.csv')

# Step 1: split into X and y
X_train = train_scaled.drop(columns=[target_col])
y_train = train_scaled[target_col]

X_val = val_scaled.drop(columns=[target_col])
y_val = val_scaled[target_col]

# Step 2: train logistic regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 3: predict probabilities
val_probs = model.predict_proba(X_val)[:, 1]

# Step 4: score it
auc = roc_auc_score(y_val, val_probs)
print("Validation ROC-AUC:", auc)


train_probs = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_probs)

print("Train ROC-AUC:", train_auc)
print("Validation ROC-AUC:", auc)