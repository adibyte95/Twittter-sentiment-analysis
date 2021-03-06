'''
This script is used to pull data from twitter
Please Note:
1. you will have to provide you own consumer_key,consumer_secret,
acess_token and access_token_secret
2. in position of #Name you will have to provide the appropriate 
hashtag.
3. hashtags are stored in tweets.csv file 
'''

import tweepy
import csv
import pandas as pd
####input your credentials here
consumer_key = 'consumer key here'
consumer_secret = 'consumer secret here'
access_token = 'access token here'
access_token_secret = 'access token secret'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('tweets.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

counter  = 0
for tweet in tweepy.Cursor(api.search,q="#Name",count=100,
                           lang="en",
                           since="2017-04-03").items():
    print('pulled tweet no: ', counter)
    print ( tweet.text)
    counter = counter + 1
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])