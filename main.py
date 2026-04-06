import pandas as pd
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# load data
df = pd.read_csv("data/dataset.csv")

X = df.drop("default", axis=1)
y = df["default"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# model
model = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    scale_pos_weight=3,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# predictions
y_prob = model.predict_proba(X_test)[:, 1]

# evaluation
roc = roc_auc_score(y_test, y_prob)
print("ROC AUC:", roc)

# save model
joblib.dump(model, "models/credit_model.pkl")
