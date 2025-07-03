"""
This module defines the FastAPI application for the Clarion project.

It includes the API endpoints for making predictions with the Titanic survival model,
as well as the logic for loading and managing the machine learning model.
"""
#pylance: basic
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from . import utils
from contextlib import asynccontextmanager

class PassengerData(BaseModel):
    """
    Pydantic model for a single passenger's data.
    """
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

class ModelService:
    """
    A service class for loading and using the machine learning model.
    """
    def __init__(self, model_path: str = '/tmp/model.pkl', scaler_path: str = '/tmp/scaler.pkl', columns_path: str = '/tmp/columns.pkl'):
        """
        Initializes the ModelService by loading the model, scaler, and training columns.

        Args:
            model_path (str, optional): Path to the model file. Defaults to '/tmp/model.pkl'.
            scaler_path (str, optional): Path to the scaler file. Defaults to '/tmp/scaler.pkl'.
            columns_path (str, optional): Path to the training columns file. Defaults to '/tmp/columns.pkl'.
        """
        self.model = joblib.load(model_path) # type: ignore
        self.scaler = joblib.load(scaler_path) # type: ignore
        self.training_columns = joblib.load(columns_path) # type: ignore

    def preprocess(self, data: PassengerData) -> np.ndarray:
        """
        Preprocesses the passenger data to be used by the model.

        Args:
            data (PassengerData): The passenger data to preprocess.

        Returns:
            np.ndarray: The preprocessed data as a NumPy array.
        """
        df = pd.DataFrame([data.model_dump()])
        df['Is_Alone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)
        df_encoded = pd.get_dummies(data=df, columns=["Sex", "Embarked"])
        df_realigned = df_encoded.reindex(columns=self.training_columns, fill_value=0) # type: ignore
        df_scaled = self.scaler.transform(df_realigned) # type: ignore
        return df_scaled # type: ignore

    def predict(self, data: np.ndarray) -> int:
        """
        Makes a prediction using the loaded model.

        Args:
            data (np.ndarray): The preprocessed data to make a prediction on.

        Returns:
            int: The prediction (0 for not survived, 1 for survived).
        """
        prediction = self.model.predict(data) # type: ignore
        return int(prediction[0]) # type: ignore

async def download_models():
    """
    Downloads the model files from MinIO.
    """
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/titanic_model.pkl', local_path='/tmp/model.pkl')
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/titanic_scaler.pkl', local_path='/tmp/scaler.pkl')
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/training_columns.pkl', local_path='/tmp/columns.pkl')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the application's lifespan.

    This function is responsible for loading the machine learning model when the application starts
    and cleaning up the resources when the application shuts down.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Load the ML model
    await download_models()
    app.state.model_service = ModelService()
    yield
    # Clean up the ML models and release the resources
    app.state.model_service = None

app = FastAPI(lifespan=lifespan)

def get_model_service():
    """
    Returns the instance of the ModelService.

    This is used as a dependency in the predict endpoint.

    Returns:
        ModelService: The instance of the ModelService.
    """
    return app.state.model_service

@app.post("/predict")
async def predict(data: PassengerData, model_service: ModelService = Depends(get_model_service)):
    """
    API endpoint for making predictions.

    Args:
        data (PassengerData): The passenger data to make a prediction for.
        model_service (ModelService, optional): The model service. Defaults to Depends(get_model_service).

    Returns:
        dict: A dictionary containing the prediction.
    """
    df_scaled = model_service.preprocess(data)
    prediction = model_service.predict(df_scaled)
    return {"Survived": prediction}