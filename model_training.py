import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load the dataset
data = pd.read_csv("train.csv")

# Drop unnecessary columns (adjust based on your dataset)
data = data.drop(columns=["Unnamed: 0"])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Use TfidfVectorizer to convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model on the test set
y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=["sadness", "joy", "love", "anger", "fear", "surprise"]))

# Save the model and vectorizer for future use
with open("emotion_model.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)
