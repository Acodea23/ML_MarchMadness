
# Research Question

When two teams face each other in the NCAA Men's Basketball Tournament, who will win the game? Can we consistently predict the winner using available season statistics and tournament matchup data?

### Dataset Overview

Our analysis uses two publicly scraped datasets from Sports Reference:

- `teams.csv`: yearly team statistics for each NCAA Division I squad, including wins, losses, win percentage, strength of schedule (SRS), shooting efficiency (FG%, 3P%, FT%), rebounds, assists, turnovers, and scoring margins.
- `tournament_games.csv`: game-level NCAA tournament results listing the winner and loser team names, their tournament seeds, and the season.

These two raw datasets are merged in `cleandata.ipynb` by matching each tournament game's winning and losing team to their season statistics. The merged dataset then computes matchup difference features such as `SRS_diff`, `win_pct_diff`, `FG%_diff`, `3P%_diff`, `FT%_diff`, `TRB_diff`, `AST_diff`, and `TOV_diff`. The final modeling dataset is saved as `data/tournament_model_ml.csv`.

The final dataset is structured so that each row compares two teams from the same game: the winner's season stats minus the loser's season stats. The target variable `win_label` is then created by using both perspectives of each game (winner perspective labeled `1` and loser perspective labeled `0`) to support binary classification.

### Source Legitimacy

The data come from Sports Reference’s public NCAA men's basketball pages (`https://www.sports-reference.com/cbb/`). Only public team and game statistics were collected, with no personal or private data included. Scraping was performed responsibly for an academic project, with rate limiting and no attempt to overwhelm the Sports Reference servers.

# Data Description and Variables

Our final dataset that will be used for our models contains the season that the game occured. It also contains the difference in various statistics for the teams competing. Calculating the difference between the teams calculated by the winner's statistics minus the loser's statistics. It is titled *tournament_model_ml.csv*.

## Key Variables
Our target variable is **win-label**, which labels which team won the game. A 1 means the first team won, a 0 means the team who's stats were subtracted won. 

 The difference for the following stats:
    - 3p%: The percentage of 3 point attempts made by a team.
    - AST: An assist is a pass that leads to a made basket.
    - FG% This is the percentage of all field goal attempts that are made.
    - FT%: This is the percentage of all free throw shots that are made.
    - SRS: A rating that takes into account average point differential and strength of schedule. The rating is denominated in points above/below average, where zero is average. Non-Division I games are excluded from the ratings.
    - TOV: Turnovers, when the ball is given to the other team without a shot attempt.
    - TRB: Total Rebounds, when a shot is missed and the team grabs the ball this is a rebound.
    - Seed: How the tournament organizes the teams. Lower is better. They are 1-16.
    - win-pct: The percentage of games a team won.
    
## Preprocessing Steps
- Missing-value handling: team stat columns were coerced to numeric using `pd.to_numeric(..., errors="coerce")`, and the final ML dataset dropped any rows with missing values after merging winner and loser stats using `final_df = final_df.dropna()`.
- Duplicate removal: accidental repeated header rows from scraped CSVs were removed by filtering out rows where `team == "School"`; the final dataset was also built with explicit winner/loser perspectives and unique difference features to avoid duplicate columns.
- Column renaming/filtering: scraped data was normalized by cleaning team names, dropping unwanted ranking columns like `Rk` and `Rank`, removing separator columns matching `^Unnamed`, and renaming fields to consistent labels such as `School` → `team`, `W-L%` → `win_pct`, `PS/G` → `points_for`, `PA/G` → `points_against`, plus repeated column renames for duplicate win/loss fields in the team stats export.

# Summary Statistics

## Numeric Variables
All numeric variables have 1,068 observations, indicating a complete dataset with no missing values for these features.

Because all variables represent differences between two teams, the means and medians are exactly zero (or numerically indistinguishable from zero). This is expected and confirms that the dataset is properly constructed: for every matchup, advantages on one side are balanced by disadvantages on the other.

Shooting percentage differences (3P%_diff, FG%_diff, FT%_diff) exhibit relatively small variability:

- Typical differences are within ±2–4 percentage points (interquartile ranges),
- Extreme values reach roughly ±10–16 percentage points, indicating occasional large shooting mismatches.

Volume statistics show much larger dispersion:

- AST_diff (assists) has a standard deviation of ~108 with ranges exceeding ±350,
- TRB_diff (rebounds) has the widest spread, ranging from −680 to +680,
- TOV_diff (turnovers) also varies substantially, reflecting the cumulative nature of these stats.

Team strength indicators show meaningful variability:

- SRS_diff ranges from −38.6 to +38.6, indicating matchups between very uneven teams,
- seed_diff spans the full tournament range (−15 to +15),
- win_pct_diff ranges from −0.488 to +0.488, capturing large regular‑season performance gaps.

The dataset combines low‑variance efficiency metrics (shooting percentages) with high‑variance cumulative and strength metrics (rebounds, assists, SRS, seeds), suggesting that different types of variables may contribute differently to predicting wins.


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3P%_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>0.035930</td>
      <td>-0.105</td>
      <td>-0.02400</td>
      <td>0.0</td>
      <td>0.02400</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>AST_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>107.942364</td>
      <td>-366.000</td>
      <td>-74.25000</td>
      <td>0.0</td>
      <td>74.25000</td>
      <td>366.000</td>
    </tr>
    <tr>
      <th>FG%_diff</th>
      <td>1068.0</td>
      <td>4.158139e-19</td>
      <td>0.033422</td>
      <td>-0.122</td>
      <td>-0.02200</td>
      <td>-0.0</td>
      <td>0.02200</td>
      <td>0.122</td>
    </tr>
    <tr>
      <th>FT%_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>0.047550</td>
      <td>-0.163</td>
      <td>-0.03225</td>
      <td>0.0</td>
      <td>0.03225</td>
      <td>0.163</td>
    </tr>
    <tr>
      <th>SRS_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>12.358625</td>
      <td>-38.630</td>
      <td>-7.41250</td>
      <td>0.0</td>
      <td>7.41250</td>
      <td>38.630</td>
    </tr>
    <tr>
      <th>TOV_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>64.969187</td>
      <td>-239.000</td>
      <td>-44.00000</td>
      <td>0.0</td>
      <td>44.00000</td>
      <td>239.000</td>
    </tr>
    <tr>
      <th>TRB_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>172.544163</td>
      <td>-680.000</td>
      <td>-117.25000</td>
      <td>-0.0</td>
      <td>117.25000</td>
      <td>680.000</td>
    </tr>
    <tr>
      <th>seed_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>7.848137</td>
      <td>-15.000</td>
      <td>-7.00000</td>
      <td>0.0</td>
      <td>7.00000</td>
      <td>15.000</td>
    </tr>
    <tr>
      <th>win_pct_diff</th>
      <td>1068.0</td>
      <td>0.000000e+00</td>
      <td>0.147874</td>
      <td>-0.488</td>
      <td>-0.10100</td>
      <td>0.0</td>
      <td>0.10100</td>
      <td>0.488</td>
    </tr>
  </tbody>
</table>
</div>

## Categorical Variables
win_label

Binary outcome variable:
- 1 = win
- 0 = loss

Because each matchup is represented from both perspectives, the dataset is balanced (534 wins and losses) by construction, making it suitable for classification without concern for outcome imbalance.

season

Observations span multiple tournament seasons.
This enables either:
- pooled modeling across years, or
- future extensions such as season‑specific analysis or temporal validation.

The categorical structure aligns well with modeling goals: a clean binary target variable and a time identifier for potential robustness checks.

## Correlation Matrix 

The correlation matrix reveals several notable relationships among predictors:
Strong positive correlations

- FG%_diff with AST_diff (0.61) and 3P%_diff (0.58):
Teams that shoot better also tend to move the ball well.

- SRS_diff with AST_diff (0.56) and TRB_diff (0.56):
Stronger teams generally outperform weaker opponents across multiple statistics.

- win_pct_diff with FG%_diff, AST_diff, and SRS_diff (≈0.50–0.55):
Season‑long success aligns with game‑level advantages.

Strong negative correlations

- seed_diff with SRS_diff (−0.93):
This extremely strong relationship confirms that tournament seeds largely reflect team quality as measured by SRS.

- seed_diff with performance metrics such as AST_diff and TRB_diff (≈ −0.50):
Higher‑seeded teams tend to dominate on the stat sheet.

Weaker or near‑zero relationships

- TOV_diff is only weakly correlated with most variables, suggesting turnovers may play a more situational or opponent‑dependent role.


There is moderate to strong multicollinearity among measures of team quality and performance, particularly among SRS_diff, seed_diff, and win_pct_diff. This should be considered during modeling.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>3P%_diff</th>
      <th>AST_diff</th>
      <th>FG%_diff</th>
      <th>FT%_diff</th>
      <th>SRS_diff</th>
      <th>TOV_diff</th>
      <th>TRB_diff</th>
      <th>seed_diff</th>
      <th>win_pct_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3P%_diff</th>
      <td>1.000000</td>
      <td>0.377422</td>
      <td>0.582878</td>
      <td>0.271383</td>
      <td>0.217466</td>
      <td>-0.188244</td>
      <td>-0.027122</td>
      <td>-0.182182</td>
      <td>0.302916</td>
    </tr>
    <tr>
      <th>AST_diff</th>
      <td>0.377422</td>
      <td>1.000000</td>
      <td>0.606446</td>
      <td>0.137013</td>
      <td>0.561088</td>
      <td>0.149373</td>
      <td>0.538355</td>
      <td>-0.495156</td>
      <td>0.504211</td>
    </tr>
    <tr>
      <th>FG%_diff</th>
      <td>0.582878</td>
      <td>0.606446</td>
      <td>1.000000</td>
      <td>0.206220</td>
      <td>0.347916</td>
      <td>-0.063254</td>
      <td>0.156278</td>
      <td>-0.291044</td>
      <td>0.519371</td>
    </tr>
    <tr>
      <th>FT%_diff</th>
      <td>0.271383</td>
      <td>0.137013</td>
      <td>0.206220</td>
      <td>1.000000</td>
      <td>0.208818</td>
      <td>-0.217719</td>
      <td>-0.085450</td>
      <td>-0.190244</td>
      <td>0.128055</td>
    </tr>
    <tr>
      <th>SRS_diff</th>
      <td>0.217466</td>
      <td>0.561088</td>
      <td>0.347916</td>
      <td>0.208818</td>
      <td>1.000000</td>
      <td>-0.034144</td>
      <td>0.559851</td>
      <td>-0.933950</td>
      <td>0.547559</td>
    </tr>
    <tr>
      <th>TOV_diff</th>
      <td>-0.188244</td>
      <td>0.149373</td>
      <td>-0.063254</td>
      <td>-0.217719</td>
      <td>-0.034144</td>
      <td>1.000000</td>
      <td>0.404907</td>
      <td>0.009216</td>
      <td>-0.119918</td>
    </tr>
    <tr>
      <th>TRB_diff</th>
      <td>-0.027122</td>
      <td>0.538355</td>
      <td>0.156278</td>
      <td>-0.085450</td>
      <td>0.559851</td>
      <td>0.404907</td>
      <td>1.000000</td>
      <td>-0.523486</td>
      <td>0.461096</td>
    </tr>
    <tr>
      <th>seed_diff</th>
      <td>-0.182182</td>
      <td>-0.495156</td>
      <td>-0.291044</td>
      <td>-0.190244</td>
      <td>-0.933950</td>
      <td>0.009216</td>
      <td>-0.523486</td>
      <td>1.000000</td>
      <td>-0.543627</td>
    </tr>
    <tr>
      <th>win_pct_diff</th>
      <td>0.302916</td>
      <td>0.504211</td>
      <td>0.519371</td>
      <td>0.128055</td>
      <td>0.547559</td>
      <td>-0.119918</td>
      <td>0.461096</td>
      <td>-0.543627</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Correlation with Winning
The table of correlations with the binary outcome identifies which variables are most strongly associated with winning:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Correlation with Win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>win_label</th>
      <td>1.000</td>
    </tr>
    <tr>
      <th>SRS_diff</th>
      <td>0.547</td>
    </tr>
    <tr>
      <th>TRB_diff</th>
      <td>0.463</td>
    </tr>
    <tr>
      <th>AST_diff</th>
      <td>0.391</td>
    </tr>
    <tr>
      <th>win_pct_diff</th>
      <td>0.354</td>
    </tr>
    <tr>
      <th>FG%_diff</th>
      <td>0.191</td>
    </tr>
    <tr>
      <th>FT%_diff</th>
      <td>0.138</td>
    </tr>
    <tr>
      <th>3P%_diff</th>
      <td>0.135</td>
    </tr>
    <tr>
      <th>TOV_diff</th>
      <td>0.109</td>
    </tr>
    <tr>
      <th>seed_diff</th>
      <td>-0.491</td>
    </tr>
  </tbody>
</table>
</div>

Because win_label is binary, these correlations are point‑biserial correlations, which measure association—not causation.

Overall team strength indicators (SRS_diff, seed_diff, win_pct_diff) dominate as predictors of winning, while shooting efficiency and turnovers provide incremental but smaller contributions.

# Visual Exploration - Issacc

![Alt text](progress/totupsets.png)
This graph shows the overall seeds with the most number of upsets, so it allows us to see what seeds seem to pull off upsets more frequently.

![Alt text](progress/statsvwins.png)
This shows the team statistics that seem to relate most to getting a win. Unsuprisingly, the teams with stronger statistics in each of the four displayed have a higher likelihood of winning according to the boxplots, but the higher 3P% is only slightly higher for the winning team than the losing team according to the boxplot.

![Alt text](progress/diffsgraphic.png)
This graphic shows much of what's displayed in the table before the graphs. The stat differential that seemed to predict a win that was highest was SRS, or strenth rating score for teams. The lowest was seed, which makes sense since better teams get a lower-numbered seed while the worse teams generally get higher-numbered seeds.

# Challenges and Reflection

 One of the first decisions was finding a reliable source of both team statistics and tournament game results. Sports Reference was chosen for its comprehensive and consistently structured historical data going back to 2015, and because no single pre-built dataset existed that combined season stats with tournament matchup outcomes in the format needed for this model. However, scraping it directly came with its own complications. The site's multi-level HTML headers caused pandas to generate duplicate and unnamed columns like Unnamed: 8_level_1, wins, wins.1, and wins.2, each representing different record splits such as conference, home, and away records, but with no clear labels out of the box. These had to be manually renamed to meaningful identifiers before the data was usable. Team name inconsistencies added another layer of difficulty, as names did not always match between the season stats and tournament game datasets. Issues like Albany (NY) versus Albany, trailing NCAA suffixes, non-breaking spaces (\xa0), and asterisks all had to be explicitly caught and cleaned, and any mismatch that slipped through would silently drop an entire game from the training data as a failed merge. Finally, rather than feeding raw stats into the model, features were engineered as difference values between the winner and loser for each game. Metrics like SRS_diff, FG%_diff, and seed_diff were computed for every matchup, and the dataset was then mirrored from the loser's perspective with all signs flipped and the label set to 0, ensuring the model saw balanced examples rather than a one-sided view where positive differences always meant a win.
