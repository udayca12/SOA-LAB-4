import mlflow.sklearn

# Replace with your actual run ID from MLflow UI
model_uri = "runs:/5a8f129265e449498d1a29ccd590cd54/sentiment_model"

# Load the model
model = mlflow.sklearn.load_model(model_uri)

# Predict sample input
sample_inputs = ["This is awesome!", "I love this product!", ]
predictions = model.predict(sample_inputs)

print("Predictions:", predictions)
