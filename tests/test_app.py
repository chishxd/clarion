"""
This module contains tests for the Clarion API application.
"""
import numpy as np
from clarion.app import ModelService, PassengerData
from fastapi.testclient import TestClient
from clarion.app import app

# Sample input data for testing
input_data = {
  "Pclass": 3,
  "Sex": "male",
  "Age": 25.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

# Create a PassengerData instance from the sample data
passenger = PassengerData(**input_data)  # type: ignore

def test_model_service_logic():
    """
    Tests the preprocessing and prediction logic of the ModelService.

    This test verifies that:
    - The preprocessing step returns a NumPy array.
    - The shape of the preprocessed data matches the number of training columns.
    - The prediction is either 0 or 1.
    """
    # Use dummy model files for testing
    service = ModelService(model_path="tests/dummy-data/model.pkl", scaler_path="tests/dummy-data/scaler.pkl", columns_path="tests/dummy-data/columns.pkl")

    # Preprocess the passenger data
    X = service.preprocess(passenger)
    # Make a prediction
    predict = service.predict(X)

    # Assert that the preprocessed data is a NumPy array
    assert isinstance(X, np.ndarray)
    # Assert that the number of features in the preprocessed data matches the training columns
    assert X.shape[1] == len(service.training_columns) # tests if the len of array returned after preprocessing is equal to length of pretrained data's columns
    # Assert that the prediction is a valid binary outcome
    assert predict in [0,1] #tests if the prediction is either 0 or 1


def test_predict_endpoint():
    """
    Tests the /predict endpoint of the FastAPI application.

    This test sends a POST request to the /predict endpoint and verifies that:
    - The HTTP status code is 200 (OK).
    - The prediction in the response is either 0 or 1.
    """
    # Use a TestClient to make requests to the FastAPI app
    with TestClient(app) as client:
        # Send a POST request to the /predict endpoint
        response = client.post('/predict', json=input_data)
        # Get the JSON response data
        data = response.json()
        # Assert that the request was successful
        assert response.status_code == 200
        # Assert that the prediction is a valid binary outcome
        assert data['Survived'] in [0,1]