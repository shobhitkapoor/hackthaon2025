import pandas as pd
import joblib

# Load model
model = joblib.load("../model/fixie_model.pkl")

# Load new tickets
new_tickets = pd.read_csv("../data/new_tickets.csv")

# Predict
predicted_fixes = model.predict(new_tickets["Customer_Description"])

# Append predictions
new_tickets["Predicted_Fix"] = predicted_fixes

# Save output
new_tickets.to_csv("../output/output_predictions.csv", index=False)
print("✅ Predictions saved to output_predictions.csv")
