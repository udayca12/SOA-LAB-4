import requests

url = "http://localhost:5000/predict"
data = {"text": "I absolutely love this!"}
response = requests.post(url, json=data)

print("Prediction:", response.json())
