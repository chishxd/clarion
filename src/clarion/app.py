#pylance: basic
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from clarion import utils
from contextlib import asynccontextmanager

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

class ModelService:
    def __init__(self, model_path: str = '/tmp/model.pkl', scaler_path: str = '/tmp/scaler.pkl', columns_path: str = '/tmp/columns.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.training_columns = joblib.load(columns_path)

    def preprocess(self, data: PassengerData) -> np.ndarray:
        df = pd.DataFrame([data.model_dump()])
        df['Is_Alone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)
        df_encoded = pd.get_dummies(data=df, columns=["Sex", "Embarked"])
        df_realigned = df_encoded.reindex(columns=self.training_columns, fill_value=0)
        df_scaled = self.scaler.transform(df_realigned)
        return df_scaled

    def predict(self, data: np.ndarray) -> int:
        prediction = self.model.predict(data)
        return int(prediction[0])

async def download_models():
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/titanic_model.pkl', local_path='/tmp/model.pkl')
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/titanic_scaler.pkl', local_path='/tmp/scaler.pkl')
    utils.download_file_from_minio(bucket_name='autocleanse-data', object_key='models/training_columns.pkl', local_path='/tmp/columns.pkl')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    await download_models()
    app.state.model_service = ModelService()
    yield
    # Clean up the ML models and release the resources
    app.state.model_service = None

app = FastAPI(lifespan=lifespan)

def get_model_service():
    return app.state.model_service

@app.post("/predict")
async def predict(data: PassengerData, model_service: ModelService = Depends(get_model_service)):
    df_scaled = model_service.preprocess(data)
    prediction = model_service.predict(df_scaled)
    return {"Survived": prediction}