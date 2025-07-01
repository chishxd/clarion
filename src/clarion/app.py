from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd
app = FastAPI()

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    
@app.post("/predict")
async def predict(data : PassengerData):
    model = joblib.load("models/titanic_model.pkl") #type:ignore
    scaler = joblib.load("models/titanic_scaler.pkl") # type: ignore
    training_columns = joblib.load("models/training_columns.pkl")  # type: ignore

    df  = pd.DataFrame([data.model_dump()])
    df['Is_Alone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)
    df_encoded = pd.get_dummies(data=df, columns=["Sex", "Embarked"])
    df_realigned = df_encoded.reindex(columns=training_columns, fill_value=0)  # type: ignore
    df_scaled = scaler.transform(df_realigned)  # type: ignore

    pred = model.predict(df_scaled)  # type: ignore
    return {"Survived" : int(pred[0])} #type: ignore