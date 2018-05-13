# Twittter sentiment analysis
Topic - to take twitter tweets and classify the tweet as positive(reflecting positive sentiment) and negative(reflecting negative sentiment)
<br/>

## 1. DataSet
I have used a kaggle data set <a href = "https://www.kaggle.com/c/twitter-sentiment-analysis2">Click here</a><br/>
Training and Testing are done on the provided data set<br/>
data set has about 50k positive tweets and 40k negative tweets
<br/>
<img src = "https://github.com/adibyte95/Twittter-sentiment-analysis/blob/master/charts/comparison.png" alt ="pos_neg chart">
<br/>
Plot of frequency of words against the words <br/>
<img src="https://github.com/adibyte95/Twittter-sentiment-analysis/blob/master/charts/freq_words.png" alt="freq_vs_words">
<br/>
This graph follows zipf's law

## 2. Preprossing
To train a classifer first of all we will have to modify the input tweet in a format which can be given to the classifier,this step is called preprossing.<br/>
It involves several steps<br/>

## 2.1 Hashtags
a word or phrase preceded by a hash sign (#), used on social media websites and applications, especially Twitter, to identify messages on a specific topic

## 2.2 URLS
used to share link to other sites in tweets.
<br/>

## 2.3 Emoticons
Are very much used nowadays in social networking sites.they are used to represent an human expression.Currently i have removed this emojis 
however for the purpose of sentiment analysis
<br/>

## 2.4 Punctuations
<br/>

## 2.5 Repeating Character
<br/>

## 2.6 Stemming algorithms
<br/>



## 3.Features 

<br/>
## 4.Expriments 

<br/>
## 5. Results

<br/>

## Future Work
1. to use another set of features and classifiers to improve accuracy
2. to use emoji as an feature for sentiment analysis and check how it affects the accuracy of the classifier
3. to train the classifer on sentiment140 dataset and used pull_tweets.py script to pull live tweets from twitter and classify them 
    in real time.
