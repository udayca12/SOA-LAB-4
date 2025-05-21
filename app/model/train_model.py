from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import os

# Sample dataset
data = [
    ("I love this product!", 1),
    ("This is the worst!", 0),
    ("Excellent service", 1),
    ("Not good at all", 0),
    ("Totally worth it", 1),
    ("Waste of money", 0)
]

texts, labels = zip(*data)

# Create pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(texts, labels)

# Save model to same directory as script
output_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(output_dir, 'sentiment_model.pkl'))
print("Model saved.")
