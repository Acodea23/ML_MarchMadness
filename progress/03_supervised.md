Baseline Mod - Matthew
Logistic Regression - Isaac
Random Forrest - Corbin

most important features
forest
SRS diff,
double others,
TRB,seed,AST


# 1. Problem Context and Research Question
When two teams face each other in the NCAA Men's Basketball Tournament, who will win the game? Our goal is to consistintly predict the winner using available season statistics and tournament matchup data?


# 2. Supervised Models Implemented
Create a table or concise summary listing:
Model type
Key hyperparameters explored
Validation setup used (for example: train/validation split or cross-validation)
Performance metrics used and final values

| Model Type | Key Hyperparameters Explored | Validation Setup | Performance Metrics & Final Values |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C` (0.001 to 100.0)<br>`penalty` ('l1', 'l2')<br>`solver` ('liblinear', 'saga') | **Train/Test Split:** Time-based (Train: 2015–2022, Test: 2023–2025)<br>**Tuning:** 5-fold Cross-Validation via `GridSearchCV` | **Metrics Targeted:** Accuracy, Classification Report, Confusion Matrix<br>**Test Accuracy:** 0.7580<br>**Precision:** 0.76<br>**Recall:** 0.76<br>**F1Score** 0.76 |
| **Random Forest Classifier** | `n_estimators` (100–800)<br>`max_depth` (None, 5–30)<br>`min_samples_split` (2–19)<br>`min_samples_leaf` (1–7)<br>`max_features` ('sqrt', 'log2', None)<br>`bootstrap` (True, False) | **Train/Test Split:** Time-based (Train: <= 2022, Test: >= 2023)<br>**Tuning:** 5-fold Cross-Validation via `RandomizedSearchCV` (50 iterations) | **Best CV AUC:** 0.7480<br>**Test Accuracy:** 0.7675<br>**Test ROC-AUC:** 0.8419 |
| **Baseline Heuristic** | **None** (Rule-based: team with the lower `seed_diff` wins; `SRS_diff` used as tiebreaker) | **Evaluation:** Tested directly on the test set (Seasons > 2022) without prior training/tuning | **Test Accuracy:** 0.7070 |


# 3. Model Comparison and Selection

As shown above, we compared three different model. Our baseline model used the teams' seeds to determine which team to predict as a winner. In the rare case of a tie it used SRS, which is a ranking system that Basketball Reference developed. We used these because they are easily accessible metrics that generally define the overall better team.
Our other two models, a logistic regression model and a random forest model both performed better than the baseline model. On our testing data, the logistic regressino model predicted 3 more games correct than the baseline model, and the random forest model predicted about 4.5 more games correctly than the baseline on average. Keep in mind, this is considering that 63 games are predicted a year.
We scaled the features for use in the logistic regression model so that our coefficients could be more easily interpreted and compared. We did not need to do this for our random forest model as it is not sensitive to scaling.
We saw that for both models, using coefficient value, and feature importance metrics, SRS was by far the most important metric in predicting the winner, and then seed and turnovers also help. This helped us to know that our baseline choices were good metrics to choose, but also that improvements from the baseline would be marginal. Thus we feel pretty good about the improvements from the baseline that we did see in our random forest model.
The random forest model performed best, we determined this by model accuracy, due to the balanced nature of our data and the equal cost of a false positive and false negative, accuracy is a good metric. In the future, we could consider including a tag for teams that advanced very far in the tournament and include an increased penalty for misclassifying games they played. We believe the random forest model performed best due to the flexibility of the model. There are probably situations, where certain features have a different impact on the outcome of a game. Having a random forest model with a depth greater than one allows for this feature flexibility.
One challenge we faced, is determing how many years of tournament games to use. Our end goal is to be able to predict winners of current tournaments, so we want to make sure that our model learns patterns in more recent years. Pulling data from older years provide us with more training data, but may cause the model to train itself on patterns that have since shifted. Finding a balance between enough data and recent data is important. We also had to tune hyperparameters but that was not too difficult.


# 4. Explainability and Interpretability
To interpret our random forest model's behavior, we extracted feature importances using scikit-learn's built-in feature_importances_ attribute on the best estimator returned by RandomizedSearchCV. This metric reflects each feature's average reduction in Gini impurity across all splits and all trees in the ensemble. Features used more often and higher in the trees receive greater importance scores.

The results show that SRS differential dominates all other features with an importance of 0.504, meaning it alone accounts for over half of the model's splitting decisions. This is consistent with what SRS measures: a team's average point margin adjusted for strength of schedule, making it a rich composite signal of overall team quality. The model effectively learned what coaches and analysts already know, that SRS is among the most predictive single-number summaries of a team.

The remaining features contributed more modestly. Rebound differential (0.093) and assist differential (0.088) were next, capturing possession control and ball movement. Win percentage differential (0.067) and free throw percentage differential (0.061) followed, with shooting efficiency metrics, turnovers, and seed rounding out the list.

Interestingly, seed differential ranked last at 0.039, despite being the backbone of our baseline model. This suggests that seeding, while a useful heuristic, is largely a coarser summary of information the model can already extract more precisely from raw statistics. The random forest effectively deprioritized seed once it had access to the underlying performance data.

This explainability output gives us confidence that the model is behaving sensibly and learning real basketball signal rather than spurious patterns. It also validates our baseline feature choices while explaining why the random forest improves upon them.


# 5. Final Takeaways
Summarize key insights gained from supervised learning.
Explain how this analysis answers your research question.

