import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load your historical training data (2015-2025)
print("Loading historical data...")
ml_df = pd.read_csv('data\\tournament_model_ml.csv')

# Define features (all _diff columns) and target
features = [
    '3P%_diff', 'AST_diff', 'FG%_diff', 'FT%_diff', 
    'SRS_diff', 'TOV_diff', 'TRB_diff', 'seed_diff', 'win_pct_diff'
]
target = 'win_label'

# 2. Train the Random Forest Model
print("Training Random Forest model on historical data...")
X_train = ml_df[features]
y_train = ml_df[target]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=7)
rf_model.fit(X_train, y_train)

# 3. Load the 2026 data
print("Loading 2026 team and tournament data...")
teams_2026 = pd.read_csv('teams_2026.csv')
games_2026 = pd.read_csv('tournament_games_2026.csv') # Or filter from tournament_games.csv

# Ensure numeric columns are formatted correctly
numeric_cols = ['AST', 'FG%', 'FT%', 'SRS', 'TOV', 'TRB', '3P%', 'win_pct']
for col in numeric_cols:
    if col in teams_2026.columns:
        teams_2026[col] = pd.to_numeric(teams_2026[col], errors='coerce')

# 4. Generate matchups and predict 2026 games
print("\nPredicting 2026 Tournament Games:\n" + "="*40)

predictions_log = []

for _, game in games_2026.iterrows():
    # Extract the two teams
    t1 = game['winner'] # Team 1 (In a prediction scenario, we just evaluate the matchup)
    t2 = game['loser']  # Team 2
    
    t1_stats = teams_2026[teams_2026['team'] == t1]
    t2_stats = teams_2026[teams_2026['team'] == t2]
    
    if t1_stats.empty or t2_stats.empty:
        print(f"Skipping {t1} vs {t2} (Missing team data)")
        continue
        
    t1_stats = t1_stats.iloc[0]
    t2_stats = t2_stats.iloc[0]
    
    # Calculate the diffs from Team 1's perspective
    matchup_features = pd.DataFrame([{
        '3P%_diff': t1_stats.get('3P%', 0) - t2_stats.get('3P%', 0),
        'AST_diff': t1_stats.get('AST', 0) - t2_stats.get('AST', 0),
        'FG%_diff': t1_stats.get('FG%', 0) - t2_stats.get('FG%', 0),
        'FT%_diff': t1_stats.get('FT%', 0) - t2_stats.get('FT%', 0),
        'SRS_diff': t1_stats.get('SRS', 0) - t2_stats.get('SRS', 0),
        'TOV_diff': t1_stats.get('TOV', 0) - t2_stats.get('TOV', 0),
        'TRB_diff': t1_stats.get('TRB', 0) - t2_stats.get('TRB', 0),
        'seed_diff': game['winner_seed'] - game['loser_seed'],
        'win_pct_diff': t1_stats.get('win_pct', 0) - t2_stats.get('win_pct', 0)
    }])
    
    # Predict probability of Team 1 winning
    win_prob = rf_model.predict_proba(matchup_features)[0][1]
    
    # Determine the model's predicted winner
    predicted_winner = t1 if win_prob >= 0.5 else t2
    confidence = max(win_prob, 1 - win_prob) * 100
    
    # Check if the model got it right (assuming t1 was the actual winner)
    correct = "✅" if predicted_winner == t1 else "❌"
    
    print(f"{t1} ({game['winner_seed']}) vs {t2} ({game['loser_seed']})")
    print(f"   -> Model Predicts: {predicted_winner} ({confidence:.1f}% confidence) {correct}")
    
    predictions_log.append({
        'Matchup': f"{t1} vs {t2}",
        'Actual Winner': t1,
        'Predicted Winner': predicted_winner,
        'Confidence (%)': round(confidence, 1),
        'Correct': correct
    })

# Output overall accuracy for 2026
results_df = pd.DataFrame(predictions_log)
correct_count = len(results_df[results_df['Correct'] == "✅"])
total_games = len(results_df)

print("\n" + "="*40)
print(f"2026 Tournament Accuracy: {correct_count}/{total_games} ({(correct_count/total_games)*100:.1f}%)")