# Jigsaw-Unintended-Bias-in-Toxicity-Classification

**27th Place Solution** of the Jigsaw Unintended Bias in Toxicity Classification competition in Kaggle.

[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

![rank](img/rank.png)

-  general\_helper.py: Utilities
-  preprocessing.py:  Text preprocessing
-  eda.ipynb: Notebook for for explanatory data analysis
-  baseline\_lstm.py: Baseline bidirectional LSTM model 
-  bert.py: Bert model
-  ensemble.py: Ensemble different model for the final step

Final pipeline: Raw data -> preprocessing.py -> baseline\_lstm.py -> bert.py -> ensemble.py

