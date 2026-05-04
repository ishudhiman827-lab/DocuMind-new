import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 🔹 Sample dataset (tum isse bada bhi kar sakti ho)
data = {
    "text": [
        "Invoice for payment of 5000 INR",
        "Hospital patient medical report diagnosis",
        "Legal contract agreement between parties",
        "Bank transaction statement details",
        "Research paper on machine learning",
        "Patient admitted in hospital for surgery",
        "Invoice number 123 amount due",
        "Contract terms and conditions",
    ],
    "label": [
        "Invoice",
        "Medical",
        "Legal",
        "Finance",
        "Research",
        "Medical",
        "Invoice",
        "Legal"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# 🔥 ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# save model
joblib.dump(model, "model.pkl")