
# Research Question

When two teams face each other the the NCAA Men's Basketball Tournament, who will win the game? Can we consistently predict the winner comparing available statistics of the two teams.

### Dataset Overview

Our data can be considered in two parts

 - Yearly Team Statistics: This dataset provides us with yearly statistics about each team in the NCAA Tournament. It includes statistics including but not limited to - Wins, Losses, Win Percentage, Strength of Schedule, Field Goal Percentage and number made, Free Throw Percentage and number made, turnovers, and total rebounds.

- Tournament Dataset: This is the dataset we will be using for our modeling and predictions. It provides us with each NCAA tournament game for every year, and the winner of that game. If the lower seeded team won the game, the winner is a 1, if not the winner is a 0. If the teams are the same seed, the team with a higher win percetage is chosen to represent the "lower seeded" team. Also included are columns that provide the difference in the teams' seeds, win percentage, free throw percentage, total rebounds, among other things as described in the yearly team statisctics dataset. For each year, there are 67 tournament games played. And 68 teams used.

### Source Legitimacy

This data is ethically scraped from Sports Reference's website. Specifically their [college basketball section](https://www.sports-reference.com/cbb/). They allow scraping from their website for projects like ours. We made sure to include a lag between each scrape to not overwhelm their servers or cause problems. Finally, this data is about team statistics and performances, so it does not violate anyone's privacy. Nor does it  have the potential to harm omeone when analyzed.