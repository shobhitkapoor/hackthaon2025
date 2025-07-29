import pandas as pd
import joblib
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load ticket data
df = pd.read_csv("../data/historical_tickets.csv")
descriptions = df["Customer_Description"]
labels = df["Fix_Applied"]

# Vectorize descriptions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(descriptions)

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_embedded = tsne.fit_transform(X.toarray())

# Plot
plt.figure(figsize=(10, 7))
unique_labels = labels.unique()

for label in unique_labels:
    idx = labels == label
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.6)

plt.legend()
plt.title("Ticket Clusters by Fix Applied")
plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/ticket_clusters.png")
plt.show()
