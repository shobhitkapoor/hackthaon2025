import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load current training data and feedback
df_historical = pd.read_csv("../data/historical_tickets.csv")
df_feedback = pd.read_csv("../data/feedback.csv")  # feedback.csv should contain the same columns

# Combine both datasets
df_combined = pd.concat([df_historical, df_feedback], ignore_index=True)

X = df_combined["Customer_Description"]
y = df_combined["Fix_Applied"]

# Re-train model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline.fit(X, y)

# Save updated model
joblib.dump(pipeline, "../model/fixie_model.pkl")
print("✅ Model updated with new feedback!")
