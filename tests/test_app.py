import numpy as np
from clarion.app import ModelService, PassengerData
from fastapi.testclient import TestClient
from clarion.app import app

input_data = {
  "Pclass": 3,
  "Sex": "male",
  "Age": 25.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

passenger = PassengerData(**input_data)  # type: ignore

def test_preprocess_returns_expected_shape():
    # Use your real .pkl files for now (they should exist in /tmp)
    service = ModelService(model_path="tests/dummy-data/model.pkl", scaler_path="tests/dummy-data/scaler.pkl", columns_path="tests/dummy-data/columns.pkl")

    X = service.preprocess(passenger)
    predict = service.predict(X)

    assert isinstance(X, np.ndarray)
    assert X.shape[1] == len(service.training_columns) # tests if the len of array returned after preprocessing is equal to length of pretrained data's columns
    assert predict in [0,1] #tests if the prediction is either 0 or 1


def test_predict():
    with TestClient(app) as client:
        response = client.post('/predict', json=input_data)
        data = response.json()
        assert response.status_code == 200
        assert data['Survived'] in [0,1]