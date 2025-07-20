
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
