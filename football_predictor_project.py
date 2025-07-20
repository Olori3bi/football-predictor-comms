# Directory Structure Explanation

football-score-predictor/
│
├── data/
│   └── raw_matches.csv
│
├── model/
│   ├── train_model.py
│   └── score_predictor.pkl
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   └── style.css
│
├── .gitignore
└── README.md

# Sample Files Content

# data/raw_matches.csv
HomeTeam,AwayTeam,FTR
Chelsea,Arsenal,H
Liverpool,Man City,D
Everton,Man Utd,A

# model/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('../data/raw_matches.csv')
df = df[['HomeTeam', 'AwayTeam', 'FTR']]
df['FTR'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
team_mapping = {team: idx for idx, team in enumerate(teams)}
df['HomeTeam'] = df['HomeTeam'].map(team_mapping)
df['AwayTeam'] = df['AwayTeam'].map(team_mapping)

X = df[['HomeTeam', 'AwayTeam']]
y = df['FTR']
model = RandomForestClassifier()
model.fit(X, y)

with open('score_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

# backend/main.py
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

# backend/requirements.txt
fastapi
uvicorn
scikit-learn
pandas

# frontend/index.html
<!DOCTYPE html>
<html>
<head>
    <title>Football Score Predictor</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Football Score Predictor</h1>
    <label>Home Team ID:</label>
    <input id="homeTeam" type="number"><br>
    <label>Away Team ID:</label>
    <input id="awayTeam" type="number"><br>
    <button onclick="predict()">Predict Result</button>
    <p id="result"></p>
    <script>
        async function predict() {
            const home = document.getElementById("homeTeam").value;
            const away = document.getElementById("awayTeam").value;
            const res = await fetch('YOUR_BACKEND_URL/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ home_team: parseInt(home), away_team: parseInt(away) })
            });
            const data = await res.json();
            document.getElementById("result").innerText = "Predicted: " + data.prediction;
        }
    </script>
</body>
</html>

# frontend/style.css
body {
    font-family: Arial;
    margin: 20px;
}

input, button {
    margin: 10px 0;
    padding: 5px;
}

# README.md
# Football Score Predictor App

## Steps to Run Locally:
1. Train Model: `python model/train_model.py`
2. Run Backend API: `uvicorn main:app --reload` in backend folder
3. Open frontend/index.html in browser

## Deployment Steps:
Backend on Render, Frontend on Netlify.

## Manual GitHub Upload:
1. Go to GitHub -> New Repository
2. Upload folders using 'Add File' -> 'Upload Files'
