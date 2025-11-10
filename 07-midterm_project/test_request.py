import requests

url = "http://localhost:9696/predict"

person = {
    "age": 31,
    "workclass": "Private",
    "fnlwgt": 45781,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 7500,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}

response = requests.post(url, json=person)
print(response.json())
