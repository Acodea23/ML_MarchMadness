
# Research Question - Matthew

When two teams face each other the the NCAA Men's Basketball Tournament, who will win the game? Can we consistently predict the winner comparing available statistics of the two teams.

### Dataset Overview

Our data can be considered in two parts

 - Yearly Team Statistics: This dataset provides us with yearly statistics about each team in the NCAA Tournament. It includes statistics including but not limited to - Wins, Losses, Win Percentage, Strength of Schedule, Field Goal Percentage and number made, Free Throw Percentage and number made, turnovers, and total rebounds.

- Tournament Dataset: This is the dataset we will be using for our modeling and predictions. It provides us with each NCAA tournament game for every year, and the winner of that game. If the lower seeded team won the game, the winner is a 1, if not the winner is a 0. If the teams are the same seed, the team with a higher win percetage is chosen to represent the "lower seeded" team. Also included are columns that provide the difference in the teams' seeds, win percentage, free throw percentage, total rebounds, among other things as described in the yearly team statisctics dataset. For each year, there are 67 tournament games played. And 68 teams used.

### Source Legitimacy

This data is ethically scraped from Sports Reference's website. Specifically their [college basketball section](https://www.sports-reference.com/cbb/). They allow scraping from their website for projects like ours. We made sure to include a lag between each scrape to not overwhelm their servers or cause problems. Finally, this data is about team statistics and performances, so it does not violate anyone's privacy. Nor does it  have the potential to harm omeone when analyzed.

# Data Description and Variables - Matthew

## Key Variables
List target variable(s)
List predictor/feature variables

## Preprocessing Steps
Note any missing-value handling
Note any duplicates removed
Note any column renaming or filtering

# Summary Statistics - Corbin

## Numeric Variables
Placeholder for sample size, mean, standard deviation, or five-number summary

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

## Categorical Variables
Placeholder for sample size and category counts

## Correlation Matrix (if applicable)
Placeholder for correlation matrix
Notes on interesting relationships
Short interpretation highlighting key insights

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

# Visual Exploration - Issacc

Visualization 1
Placeholder for figure
Explanation: what it shows & why it’s relevant

Visualization 2
Placeholder for figure
Explanation: what it shows & why it’s relevant

# Challenges and Reflection - Corbin

Challenges faced in dataset selection or preprocessing
Concerns or challenges currently facing in the project
Short reflection/insight
