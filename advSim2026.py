import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load ADVANCED data and train the model
print("Training Advanced Random Forest Model...")
try:
    ml_df = pd.read_csv('tournament_model_adv_ml.csv') 
except FileNotFoundError:
    print("ERROR: Please move 'tournament_model_adv_ml.csv' into this folder!")
    exit()

# List of core metrics
base_metrics = ['adj_off_eff', 'adj_def_eff', 'tempo', 'efg_pct', 'tov_pct', 'orb_pct', 'ft_rate']

# Check columns and create diffs only if necessary
for metric in base_metrics:
    diff_col = f'{metric}_diff'
    
    # If the diff column isn't there, try to create it
    if diff_col not in ml_df.columns:
        col1, col2 = f'{metric}_1', f'{metric}_2'
        
        if col1 in ml_df.columns and col2 in ml_df.columns:
            ml_df[diff_col] = ml_df[col1] - ml_df[col2]
        else:
            # If we get here, the column names in your CSV are different
            print(f"\nERROR: Could not find '{diff_col}' OR '{col1}'/'{col2}' in your CSV.")
            print("Available columns in your file are:")
            print(list(ml_df.columns))
            exit()

# Final feature list
features = [f'{m}_diff' for m in base_metrics] + ['seed_diff', 'win_pct_diff']

# Clean and Train
ml_df = ml_df.dropna(subset=features + ['win_label'])
rf_model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
rf_model.fit(ml_df[features], ml_df['win_label'])

# 2. Load the 2026 Team Stats
print("Loading 2026 Advanced Team Data...")
try:
    teams_2026 = pd.read_csv('teams_2026.csv')
except FileNotFoundError:
    print("ERROR: Could not find teams_2026.csv")
    exit()

# Ensure 2026 columns are numeric
cols_to_fix = base_metrics + ['win_pct']
for col in cols_to_fix:
    if col in teams_2026.columns:
        teams_2026[col] = pd.to_numeric(teams_2026[col], errors='coerce')

# 3. Prediction Logic
def predict_game(t1, seed1, t2, seed2):
    t1_stats = teams_2026[teams_2026['team'] == t1]
    t2_stats = teams_2026[teams_2026['team'] == t2]
    
    if t1_stats.empty or t2_stats.empty:
        missing = t1 if t1_stats.empty else t2
        print(f"  [!] Missing Advanced Data for {missing}. Defaulting to Seed.")
        return (t1, seed1, 1.0) if seed1 <= seed2 else (t2, seed2, 1.0)
        
    t1_s = t1_stats.iloc[0]
    t2_s = t2_stats.iloc[0]
    
    # Calculate the "Diffs" for the 2026 matchup
    matchup_data = {}
    for m in base_metrics:
        matchup_data[f'{m}_diff'] = t1_s.get(m, 0) - t2_s.get(m, 0)
    
    matchup_data['seed_diff'] = seed1 - seed2
    matchup_data['win_pct_diff'] = t1_s.get('win_pct', 0) - t2_s.get('win_pct', 0)
    
    matchup = pd.DataFrame([matchup_data])
    
    # Reorder columns to match the model's training order
    matchup = matchup[features]
    
    prob = rf_model.predict_proba(matchup)[0][1]
    return (t1, seed1, prob) if prob >= 0.5 else (t2, seed2, 1 - prob)

# 4. The Official 2026 Starting Bracket
matchups = [
    # --- EAST REGION ---
    [("Duke", 1), ("Siena", 16)], [("Ohio State", 8), ("TCU", 9)],
    [("St. John's", 5), ("Northern Iowa", 12)], [("Kansas", 4), ("Cal Baptist", 13)],
    [("Louisville", 6), ("South Florida", 11)], [("Michigan State", 3), ("North Dakota State", 14)],
    [("UCLA", 7), ("UCF", 10)], [("UConn", 2), ("Furman", 15)],
    # --- WEST REGION ---
    [("Arizona", 1), ("Long Island", 16)], [("Villanova", 8), ("Utah State", 9)],
    [("Wisconsin", 5), ("High Point", 12)], [("Arkansas", 4), ("Hawaii", 13)],
    [("BYU", 6), ("Texas", 11)], [("Gonzaga", 3), ("Kennesaw State", 14)],
    [("Miami (FL)", 7), ("Missouri", 10)], [("Purdue", 2), ("Queens", 15)],
    # --- SOUTH REGION ---
    [("Florida", 1), ("Lehigh", 16)], [("Clemson", 8), ("Iowa", 9)],
    [("Vanderbilt", 5), ("McNeese", 12)], [("Nebraska", 4), ("Troy", 13)],
    [("North Carolina", 6), ("VCU", 11)], [("Illinois", 3), ("Penn", 14)],
    [("Saint Mary's", 7), ("Texas A&M", 10)], [("Houston", 2), ("Idaho", 15)],
    # --- MIDWEST REGION ---
    [("Michigan", 1), ("UMBC", 16)], [("Georgia", 8), ("Saint Louis", 9)],
    [("Texas Tech", 5), ("Akron", 12)], [("Alabama", 4), ("Hofstra", 13)],
    [("Tennessee", 6), ("SMU", 11)], [("Virginia", 3), ("Wright State", 14)],
    [("Kentucky", 7), ("Santa Clara", 10)], [("Iowa State", 2), ("Tennessee State", 15)]
]

# 5. Simulation Engine
round_names = ["ROUND OF 64", "ROUND OF 32", "SWEET SIXTEEN", "ELITE EIGHT", "FINAL FOUR", "NATIONAL CHAMPIONSHIP"]
round_idx = 0

print("\n🚀 RUNNING ADVANCED 2026 SIMULATION 🚀\n" + "="*50)

while len(matchups) > 0:
    print(f"\n--- {round_names[round_idx]} ---")
    next_round = []
    for i in range(0, len(matchups), 2):
        m1 = matchups[i]
        w1, s1, p1 = predict_game(m1[0][0], m1[0][1], m1[1][0], m1[1][1])
        print(f"  {m1[0][0]} ({m1[0][1]}) vs {m1[1][0]} ({m1[1][1]}) -> {w1} ({p1*100:.1f}%)")
        
        if i + 1 < len(matchups):
            m2 = matchups[i+1]
            w2, s2, p2 = predict_game(m2[0][0], m2[0][1], m2[1][0], m2[1][1])
            print(f"  {m2[0][0]} ({m2[0][1]}) vs {m2[1][0]} ({m2[1][1]}) -> {w2} ({p2*100:.1f}%)")
            next_round.append([ (w1, s1), (w2, s2) ])
            
    matchups = next_round
    round_idx += 1
    if round_idx == 6: break

print("\n" + "="*50 + f"\n🏆 ADVANCED MODEL CHAMPION: {w1} 🏆\n" + "="*50)