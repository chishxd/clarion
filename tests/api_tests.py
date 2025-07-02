import requests

payload = { #type:ignore
  "Pclass": 3,
  "Sex": "female",
  "Age": 22,
  "SibSp": 4,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

r = requests.post("http://localhost/predict", json=payload) #type:ignore
print(r.json())