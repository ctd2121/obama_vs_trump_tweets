import datetime
import tweepy
import csv

# Original code from http://socialmedia-class.org/twittertutorial.html

# Variables that contain user credentials to access Twitter API 
ACCESS_TOKEN = 'YOUR ACCESS TOKEN'
ACCESS_SECRET = 'YOUR ACCESS TOKEN SECRET'
CONSUMER_KEY = 'YOUR API KEY'
CONSUMER_SECRET = 'ENTER YOUR API SECRET'

auth = tweepy.auth.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
# wait_on_rate_limit = True:  will make the api automatically wait for rate limits to replenish
# wait_on_rate_limit_notify = True:   will make the api print a notification when Tweepy is waiting for rate limits to replenish

# Open/create a file to which we will append data
csvFile = open('result2.csv', 'a')

# Use csv writer to save results to a CSV file
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.user_timeline,
                           id = "realDonaldTrump",
                           since = "2014-11-08",
                           until = "2018-12-26",
                           lang = "en").items():
    tweet_date = tweet.created_at
    # Write row to CSV file
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
    print(tweet.created_at, tweet.text)

csvFile.close()
