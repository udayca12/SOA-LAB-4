import mlflow
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = [
    ("I love this product!", 1),
    ("This is the worst!", 0),
]
texts, labels = zip(*data)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

model.fit(texts, labels)

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "sentiment_model")
    mlflow.log_param("model_type", "MultinomialNB")
    mlflow.log_metric("train_size", len(texts))
    print("Model logged to MLflow.")
