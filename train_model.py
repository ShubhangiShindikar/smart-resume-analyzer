import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------- SIMPLE TRAINING DATA ----------------

texts = [
    "python machine learning data science pandas numpy statistics",
    "html css javascript react web development frontend",
    "deep learning tensorflow pytorch neural networks ai",
    "sql database data analysis power bi excel statistics",
    "django flask backend api python development server"
]

labels = [
    "Data Scientist",
    "Web Developer",
    "ML Engineer",
    "Data Analyst",
    "Backend Developer"
]

# ---------------- VECTORIZE ----------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# ---------------- MODEL ----------------

model = LogisticRegression()
model.fit(X, labels)

# ---------------- SAVE ----------------

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model created successfully ✔")