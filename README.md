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
This graph follows zipf's law. Learn more about Zipf's law <a href="https://www.youtube.com/watch?v=qnfxA_mL848">Here</a>
 
## 2. Preprocessing
To train a classifer first of all we will have to modify the input tweet in a format which can be given to the classifier,this step is called preprossing.<br/>
It involves several steps<br/>

## 2.1 Hashtags
a word or phrase preceded by a hash sign (#), used on social media websites and applications, especially Twitter, to identify messages on a specific topic

## 2.2 URLS
used to share link to other sites in tweets.
we have premanently removed links from our input text as they does not provide any information about the sentiment of the text <br/>
<br/>

## 2.3 Emoticons
Are very much used nowadays in social networking sites.they are used to represent an human expression.Currently we have removed this emojis <br/>
how much useful emojis are  for the purpose of sentiment analysis remains part of the future work
<br/>

## 2.4 Punctuations
To remove punctuations from the input text<br/>
input - Arjun said "Aditya is a god boy" <br/>
output - Arjun said Aditya is a good boy <br/>
<br/>

## 2.5 Repeating Character
To remove repeating characters from the text <br/>
input - yayyyyy ! i got the job <br/>
output - yayy ! i got the job <br/>
<br/>

## 2.6 Stemming algorithms
A stemmer for English, for example, should identify the string "cats" (and possibly "catlike", "catty" etc.) as based on the root "cat", and "stems", "stemmer", "stemming", "stemmed" as based on "stem". A stemming algorithm reduces the words "fishing", "fished", and "fisher" to the root word, "fish". On the other hand, "argue", "argued", "argues", "arguing", and "argus" reduce to the stem "argu" (illustrating the case where the stem is not itself a word or root) but "argument" and "arguments" reduce to the stem "argument".
porter stemmer is used here

<br/>

## 3.Features
we have used <br/>
unigrams <br/>
bigrams <br/>
unigrams + bigrams<br/>
unigrams + bigrams +trigrams<br/>
as features
<br/>

## 4.Expriments 
we have used three model with above mentioned features 
<br/>

## 4.1 Naive bayes Classifier
<br/>

## 4.2 Maximum Entropy Classifier
<br/>

## 4.3 XGboost
<br/>


## 5. Results

<br/>

## Future Work
1. to use another set of features and classifiers to improve accuracy
2. to use emoji as an feature for sentiment analysis and check how it affects the accuracy of the classifier
3. to train the classifer on sentiment140 dataset and used pull_tweets.py script to pull live tweets from twitter and classify them 
    in real time.
