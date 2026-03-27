## AI Ideas
### 🏥 Healthcare — Sepsis Early Warning + Patient Clustering

Supervised: Train a classifier (XGBoost or LSTM) to predict sepsis onset 6–12 hours in advance using vital signs and lab values
Second technique: Cluster patients by physiological trajectory (k-means or UMAP + HDBSCAN) to discover sepsis subtypes, then train subtype-specific classifiers — this is the key insight that elevates it beyond a Kaggle task
Dataset: MIMIC-III (requires credentialing, which itself signals seriousness)
What makes it senior-level: The clustering informs the supervised model; you're not just running both independently


### 💰 Finance — Credit Default Prediction + Anomaly Detection for Data Integrity

Supervised: Logistic regression / gradient boosting to predict loan default probability, with calibrated probabilities (not just AUC)
Second technique: Isolation Forest or Autoencoder-based anomaly detection on the feature space to flag suspicious applications before scoring them — acting as a data quality gate
Dataset: Home Credit Default Risk or LendingClub
What makes it senior-level: You're modeling the pipeline, not just the prediction — anomaly detection guards the supervised model's input distribution

## How AI Influenced Our Idea Selection
Using AI as a brainstorming tool helped us understand what separates a senior-level machine learning project from a standard academic exercise. It pushed us to think beyond single-model pipelines and consider how techniques like clustering can meaningfully inform and improve a supervised model. AI also expanded our awareness of how these methods show up in real-world and industry applications, which shaped the direction of our thinking. That exposure even sparked an additional idea of using a Random Forest model to predict game outcomes and apply it to March Madness bracket prediction.

## Final research question
Who will win March Madness Tournament games based on different teams stats available
## Candidate target variable for supervised analysis
Win v Loss in game played
## Dataset choice and backup dataset
Data from Basketball reference website. Backup - March Madness datasets created on Kaggle that we can scrape and feature engineer.
## Feasibility (time, compute, scope)
The feature engineering we want to perform will not be too complicated. We will have to be mindfull of the number of variables used as there is a limit to dataset sizes.
Compute should not be too bad either. We will focus on Men's NCAA march madness for now.
## Ethical/legal considerations
We have already checked, and Basketball Reference is ethical to scrape from. The data also does not infringe on anyone's privacy.
## Planned additional ML methods
We plan to try a random forest model as well as some neural networks. We will also try a logistic regression model.

