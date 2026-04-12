import pandas as pd
import requests
import io
import itertools

# Set User-Agent to avoid getting blocked by Sports Reference
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_teams_2026():
    print("Fetching 2026 Team Stats...")
    url = "https://www.sports-reference.com/cbb/seasons/men/2026-school-stats.html"
    r = requests.get(url, headers=HEADERS)
    
    # Read the multi-level header table
    tables = pd.read_html(io.StringIO(r.text), header=[0, 1])
    df = tables[0]
    
    # Drop intermediate header rows that Sports Reference repeats
    df = df[df.iloc[:, 0] != 'School']
    
    # Flatten multi-level columns to match your 'teams.csv' format
    cols = []
    for level_0, level_1 in df.columns:
        if "Unnamed" in level_0:
            if level_1 == "School":
                cols.append("team")
            else:
                cols.append(level_1)
        else:
            cols.append(level_1)
            
    df.columns = cols
    
    # Rename matching columns to fit your exact historical schema
    df = df.rename(columns={
        'W': 'wins', 'L': 'losses', 'W-L%': 'win_pct', 
        'Tm.': 'Tm.', 'Opp.': 'Opp.', '3P%': '3P%'
    })
    
    # Add the year column
    df['year'] = 2026
    
    # Clean up strings and return
    df['team'] = df['team'].str.replace('NCAA', '').str.strip()
    
    # Save to CSV
    df.to_csv('teams_2026.csv', index=False)
    print("Saved -> teams_2026.csv")
    return df

def get_tournament_games_2026():
    print("Fetching 2026 Tournament Games...")
    # Using the sports-reference tournament schedule
    url = "https://www.sports-reference.com/cbb/postseason/men/2026-schedule.html"
    r = requests.get(url, headers=HEADERS)
    
    try:
        tables = pd.read_html(io.StringIO(r.text))
        games = tables[0]
        
        # Clean up the schedule table
        games = games[['Date', 'Round', 'Winner', 'Score', 'Loser', 'Score.1']].dropna()
        games = games[games['Round'] != 'Round'] # Drop repeats
        
        # Extract Seeds (Usually in parentheses in the team name string, e.g., "Michigan (1)")
        def extract_seed_and_team(team_str):
            import re
            match = re.search(r'\((.*?)\)', team_str)
            seed = int(match.group(1)) if match else None
            team = re.sub(r'\(.*?\)', '', team_str).strip()
            return team, seed
            
        tourney_data = []
        for _, row in games.iterrows():
            w_team, w_seed = extract_seed_and_team(row['Winner'])
            l_team, l_seed = extract_seed_and_team(row['Loser'])
            
            tourney_data.append({
                'season': 2026,
                'winner': w_team,
                'winner_seed': w_seed,
                'loser': l_team,
                'loser_seed': l_seed
            })
            
        tourney_df = pd.DataFrame(tourney_data)
        tourney_df.to_csv('tournament_games_2026.csv', index=False)
        print("Saved -> tournament_games_2026.csv")
        return tourney_df
    except Exception as e:
        print("Could not fetch games automatically (the schedule format might have changed).")
        return pd.DataFrame()

def generate_model_diffs(teams_df, games_df):
    print("Generating Model Difference Sets...")
    # Convert necessary team columns to numeric
    numeric_cols = ['AST', 'FG%', 'FT%', 'SRS', 'TOV', 'TRB', '3P%', 'win_pct']
    for col in numeric_cols:
        if col in teams_df.columns:
            teams_df[col] = pd.to_numeric(teams_df[col], errors='coerce')
            
    # Calculate advanced metrics if they aren't directly available
    # (Assuming basic estimation for standard model based on your schema)
    
    ml_data = []
    
    for _, game in games_df.iterrows():
        # Get team stats
        w_stats = teams_df[teams_df['team'] == game['winner']]
        l_stats = teams_df[teams_df['team'] == game['loser']]
        
        if w_stats.empty or l_stats.empty:
            continue
            
        w_stats = w_stats.iloc[0]
        l_stats = l_stats.iloc[0]
        
        # Win instance (label = 1)
        ml_data.append({
            '3P%_diff': w_stats.get('3P%', 0) - l_stats.get('3P%', 0),
            'AST_diff': w_stats.get('AST', 0) - l_stats.get('AST', 0),
            'FG%_diff': w_stats.get('FG%', 0) - l_stats.get('FG%', 0),
            'FT%_diff': w_stats.get('FT%', 0) - l_stats.get('FT%', 0),
            'SRS_diff': w_stats.get('SRS', 0) - l_stats.get('SRS', 0),
            'TOV_diff': w_stats.get('TOV', 0) - l_stats.get('TOV', 0),
            'TRB_diff': w_stats.get('TRB', 0) - l_stats.get('TRB', 0),
            'seed_diff': game['winner_seed'] - game['loser_seed'],
            'win_pct_diff': w_stats.get('win_pct', 0) - l_stats.get('win_pct', 0),
            'win_label': 1,
            'season': 2026
        })
        
        # Loss instance (label = 0) to balance dataset
        ml_data.append({
            '3P%_diff': l_stats.get('3P%', 0) - w_stats.get('3P%', 0),
            'AST_diff': l_stats.get('AST', 0) - w_stats.get('AST', 0),
            'FG%_diff': l_stats.get('FG%', 0) - w_stats.get('FG%', 0),
            'FT%_diff': l_stats.get('FT%', 0) - w_stats.get('FT%', 0),
            'SRS_diff': l_stats.get('SRS', 0) - w_stats.get('SRS', 0),
            'TOV_diff': l_stats.get('TOV', 0) - w_stats.get('TOV', 0),
            'TRB_diff': l_stats.get('TRB', 0) - w_stats.get('TRB', 0),
            'seed_diff': game['loser_seed'] - game['winner_seed'],
            'win_pct_diff': l_stats.get('win_pct', 0) - w_stats.get('win_pct', 0),
            'win_label': 0,
            'season': 2026
        })

    ml_df = pd.DataFrame(ml_data)
    ml_df.to_csv('tournament_model_ml_2026.csv', index=False)
    print("Saved -> tournament_model_ml_2026.csv")
    
if __name__ == "__main__":
    t_df = get_teams_2026()
    g_df = get_tournament_games_2026()
    if not g_df.empty:
        generate_model_diffs(t_df, g_df)
    print("Done! You can now append these to your main datasets.")