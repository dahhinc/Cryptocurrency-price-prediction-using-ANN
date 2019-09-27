# Cryptocurrency-price-prediction-using-ANN
A quick guide to use attached codes:
1) Using ``` tweetscraping.py ``` fetch a year by year Twitter data by keyword e.g. "Bitcoin", get rid of special characters and perform sentiment analysis of tweets.
2) Download sample dataset ``` Bitcoin_data.csv ``` (from https://bitinfocharts.com/) and merge it by "Date" with mean compound tweet score.
3) ``` gridsearch.py ``` consists of simple RNN/LSTM based NN and grid search function for hyperparameter search. Pay attention to ``` early_stop ``` parameters to prevent overfitting or underfitting of the model. 
4) Obtained hyperhapameters insert into model in ``` ANN.py ```.

Good luck!
