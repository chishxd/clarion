import requests

payload = { #type:ignore
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

r = requests.post("http://127.0.0.1:8000/predict", json=payload) #type:ignore
print(r.json())