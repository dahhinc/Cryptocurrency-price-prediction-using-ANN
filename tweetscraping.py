import pandas as pd
import math
import numpy as np
import csv
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from twitterscraper import query_tweets
import re
import nltk
from typing import Iterator, NamedTuple, Sequence
from twitterscraper.query import query_tweets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from functools import reduce
 

file = open("E:/USER/Documents/Python Scripts/tweets2.csv","w", encoding="utf-8")
file.write("tweet_id,timestamp,text\n")
d0 = dt.date(2015,1,1)
d1 = dt.date(2015,12,31)
delta = d1-d0
for tweet in query_tweets("Bitcoin",100,d0,d1,delta.days,''):
    newtext = str(tweet.text).replace('"',"\\'")
    file.write(str(tweet.id)+","+str(tweet.timestamp)+",\""+newtext+"\"\n")
file.close()

tweet2014= pd.read_csv('E:/USER/Documents/Python Scripts/tweets1.csv', engine='python')
tweet2015= pd.read_csv('E:/USER/Documents/Python Scripts/tweets2.csv', engine='python')
tweet2016= pd.read_csv('E:/USER/Documents/Python Scripts/tweets3.csv', engine='python')
tweet2017= pd.read_csv('E:/USER/Documents/Python Scripts/tweets4.csv', engine='python')
tweet2018= pd.read_csv('E:/USER/Documents/Python Scripts/tweets5.csv', engine='python')
tweet2019= pd.read_csv('E:/USER/Documents/Python Scripts/tweets6.csv', engine='python')

data = pd.concat([tweet2014,tweet2015,tweet2016,tweet2017,tweet2018,tweet2019])

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub('#\S+','',text)
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
data = data.dropna()
testing = data.text
test_result = [] #dodelat dataframe

for t in testing:
        test_result.append(tweet_cleaner(t))#vibrosit NaN
test_result
test_result = pd.Series(test_result)

data['clear'] = test_result.values


display(data.head(10))

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

data['sentiment'] = data['text'].apply(lambda x: sid.polarity_scores(x))

data["compound"] = data.sentiment.values

tempr = np.empty(data.sentiment.shape[0], dtype="Float32")

for j in range (0 ,data.sentiment.shape[0]):
    tempr[j] = data.sentiment.iloc[j]['compound']

data["compound"] = tempr
    

data["timestamp"] = pd.to_datetime(data["timestamp"])
data["timestamp"] = pd.to_datetime(data['timestamp']).dt.to_period('D')
tweet_data = data.set_index('timestamp').resample('D').mean()

tweet_data = pd.DataFrame(tweet_data)
tweet_data['Date']= tweet_data.index.strftime('%d/%m/%Y')

#Plotting sentiment values
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
sns.distplot(data['compound'], bins=15, ax=ax)
plt.show()

#adding Bitcoin's data
Bitcoin = pd.read_csv('H:/Downloads/Telegram Desktop/Bitcoin_data.csv', engine='python')

Bitcoin= Bitcoin.merge(tweet_data, on = 'Date', how = 'left')
# ready dataset for further analysis
Bitcoin.to_csv('H:/Downloads/Telegram Desktop/Bitcoin.csv',index = False)
