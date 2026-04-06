import pandas as pd
import joblib

# load trained model
model = joblib.load("models/credit_model.pkl")

# threshold (your optimized one)
threshold = 0.45

# sample input (replace with real input)
sample = {
    "income": 50000,
    "loan_amount": 20000,
    "credit_score": 650
}

# convert to dataframe
input_df = pd.DataFrame([sample])

# prediction
prob = model.predict_proba(input_df)[0][1]
prediction = 1 if prob > threshold else 0

print("Default Probability:", prob)
print("Prediction:", "Default" if prediction == 1 else "No Default")
