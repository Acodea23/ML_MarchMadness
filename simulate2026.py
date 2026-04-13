import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load historical data and train the model
print("Training Model on Historical Matchups...")
try:
    ml_df = pd.read_csv('tournament_model_ml.csv') 
except FileNotFoundError:
    print("ERROR: Please move 'tournament_model_ml.csv' into this folder!")
    exit()

features = [
    '3P%_diff', 'AST_diff', 'FG%_diff', 'FT%_diff', 
    'SRS_diff', 'TOV_diff', 'TRB_diff', 'seed_diff', 'win_pct_diff'
]
#rf_model = RandomForestClassifier(bootstrap=True, max_depth=19, max_features=None, min_samples_leaf=6, min_samples_split=9, n_estimators=100)
#rf_model.fit(ml_df[features], ml_df['win_label'])
import joblib
rf_model = joblib.load("final_random_forest.pkl")

# 2. Load the 2026 Team Stats
print("Loading 2026 Team Data...")
try:
    teams_2026 = pd.read_csv('teams_2026.csv')
except FileNotFoundError:
    print("ERROR: Could not find teams_2026.csv")
    exit()
    
for col in ['AST', 'FG%', 'FT%', 'SRS', 'TOV', 'TRB', '3P%', 'win_pct']:
    if col in teams_2026.columns:
        teams_2026[col] = pd.to_numeric(teams_2026[col], errors='coerce')

# 3. Game Prediction Logic
def predict_game(t1, seed1, t2, seed2):
    t1_stats = teams_2026[teams_2026['team'] == t1]
    t2_stats = teams_2026[teams_2026['team'] == t2]
    
    # Fallback if a team name is misspelled or missing from the dataset
    if t1_stats.empty or t2_stats.empty:
        missing = t1 if t1_stats.empty else t2
        print(f"  [!] Missing data for {missing}. Advancing higher seed by default.")
        if seed1 <= seed2: return t1, seed1, 1.0
        else: return t2, seed2, 1.0
        
    t1_stats = t1_stats.iloc[0]
    t2_stats = t2_stats.iloc[0]
    
    matchup = pd.DataFrame([{
        '3P%_diff': t1_stats.get('3P%', 0) - t2_stats.get('3P%', 0),
        'AST_diff': t1_stats.get('AST', 0) - t2_stats.get('AST', 0),
        'FG%_diff': t1_stats.get('FG%', 0) - t2_stats.get('FG%', 0),
        'FT%_diff': t1_stats.get('FT%', 0) - t2_stats.get('FT%', 0),
        'SRS_diff': t1_stats.get('SRS', 0) - t2_stats.get('SRS', 0),
        'TOV_diff': t1_stats.get('TOV', 0) - t2_stats.get('TOV', 0),
        'TRB_diff': t1_stats.get('TRB', 0) - t2_stats.get('TRB', 0),
        'seed_diff': seed1 - seed2,
        'win_pct_diff': t1_stats.get('win_pct', 0) - t2_stats.get('win_pct', 0)
    }])
    
    prob = rf_model.predict_proba(matchup)[0][1]
    
    if prob >= 0.5:
        return t1, seed1, prob
    else:
        return t2, seed2, 1 - prob

# 4. The Authentic 2026 Starting Bracket (Round of 64)
matchups = [
    # --- EAST REGION ---
    [("Duke", 1), ("Siena", 16)],
    [("Ohio State", 8), ("TCU", 9)],
    [("St. John's", 5), ("Northern Iowa", 12)],
    [("Kansas", 4), ("Cal Baptist", 13)],
    [("Louisville", 6), ("South Florida", 11)],
    [("Michigan State", 3), ("North Dakota State", 14)],
    [("UCLA", 7), ("UCF", 10)],
    [("UConn", 2), ("Furman", 15)],

    # --- SOUTH REGION --- (Moved here so East faces South in the Final Four)
    [("Florida", 1), ("Lehigh", 16)],
    [("Clemson", 8), ("Iowa", 9)],
    [("Vanderbilt", 5), ("McNeese", 12)],
    [("Nebraska", 4), ("Troy", 13)],
    [("North Carolina", 6), ("VCU", 11)],
    [("Illinois", 3), ("Penn", 14)],
    [("Saint Mary's", 7), ("Texas A&M", 10)],
    [("Houston", 2), ("Idaho", 15)],

    # --- WEST REGION --- (Moved here so West faces Midwest in the Final Four)
    [("Arizona", 1), ("Long Island", 16)],
    [("Villanova", 8), ("Utah State", 9)],
    [("Wisconsin", 5), ("High Point", 12)],
    [("Arkansas", 4), ("Hawaii", 13)],
    [("BYU", 6), ("Texas", 11)],
    [("Gonzaga", 3), ("Kennesaw State", 14)],
    [("Miami (FL)", 7), ("Missouri", 10)],
    [("Purdue", 2), ("Queens", 15)],

    # --- MIDWEST REGION ---
    [("Michigan", 1), ("UMBC", 16)],
    [("Georgia", 8), ("Saint Louis", 9)],
    [("Texas Tech", 5), ("Akron", 12)],
    [("Alabama", 4), ("Hofstra", 13)],
    [("Tennessee", 6), ("SMU", 11)],
    [("Virginia", 3), ("Wright State", 14)],
    [("Kentucky", 7), ("Santa Clara", 10)],
    [("Iowa State", 2), ("Tennessee State", 15)]
]

# 5. Simulate the 6 Rounds
round_names = ["ROUND OF 64", "ROUND OF 32", "SWEET SIXTEEN", "ELITE EIGHT", "FINAL FOUR", "NATIONAL CHAMPIONSHIP"]
round_idx = 0

print("\n🏆 STARTING FULL 2026 TOURNAMENT SIMULATION 🏆\n" + "="*50)

while len(matchups) > 0:
    print(f"\n{'='*15} {round_names[round_idx]} {'='*15}")
    next_round = []
    
    for i in range(0, len(matchups), 2):
        # Play First Game of the pair
        m1 = matchups[i]
        w1, s1, p1 = predict_game(m1[0][0], m1[0][1], m1[1][0], m1[1][1])
        print(f"  {m1[0][0]} ({m1[0][1]}) vs {m1[1][0]} ({m1[1][1]}) -> WINNER: {w1} ({p1*100:.1f}%)")
        
        # Play Second Game of the pair
        if i + 1 < len(matchups):
            m2 = matchups[i+1]
            w2, s2, p2 = predict_game(m2[0][0], m2[0][1], m2[1][0], m2[1][1])
            print(f"  {m2[0][0]} ({m2[0][1]}) vs {m2[1][0]} ({m2[1][1]}) -> WINNER: {w2} ({p2*100:.1f}%)")
            print("  ---")
            
            # Form the new matchup for the next round
            next_round.append([ (w1, s1), (w2, s2) ])
            
    matchups = next_round
    round_idx += 1
    
    if len(matchups) == 0:
        print("\n" + "="*50)
        print(f"🎉 2026 NATIONAL CHAMPION PREDICTION: {w1} 🎉")
        print("="*50)