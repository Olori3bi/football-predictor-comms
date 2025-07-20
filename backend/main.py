
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

with open('../model/score_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

class MatchInput(BaseModel):
    home_team: int
    away_team: int

@app.post('/predict')
def predict_score(data: MatchInput):
    prediction = model.predict([[data.home_team, data.away_team]])[0]
    result = {1: "Home Win", 0: "Draw", -1: "Away Win"}
    return {"prediction": result[prediction]}
