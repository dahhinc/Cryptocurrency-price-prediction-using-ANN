# Cryptocurrency-price-prediction-using-ANN
A quick guide to use attached codes:
1) Using ``` tweetscraping.py ``` fetch a year by year Twitter data by a keyword e.g. "Bitcoin", get rid of special characters and perform sentiment analysis of the tweets.
2) Download sample dataset ``` Bitcoin_data.csv ``` (from https://bitinfocharts.com/) and merge it by "Date" with the mean compound tweet score.
3) ``` gridsearch.py ``` consists of the simple RNN/LSTM based NN and a grid search function for hyperparameter search. Pay attention to ``` early_stop ``` parameters to prevent overfitting or underfitting of the model. 
4) Insert obtained hyperhapameters into the model in ``` ANN.py ```.

Good luck!
