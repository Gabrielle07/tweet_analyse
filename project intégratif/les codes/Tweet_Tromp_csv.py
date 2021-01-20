import jsonpickle
import tweepy
import csv

# Keys and tokens from the Twitter Application
consumer_key = 'Jf6WZQs1zFjIP8QtUijNP3pUw'
consumer_secret = 'rZDdSQsNP8rKSlX6Y7RSZmeuE1IhvGJKFwKIlg4qNT4tErg2oW'
access_token = '1313915263608336384-1pH1er3KeFqc7grFYPbTgcavr8rgdb'
access_token_secret = '8xwGvAjkpHrv9XAqJHeJWugdG8e26tjklaSlJuBh2WZSP'

# Pass our consumer key and consumer secret to Tweepy's user authentication handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Pass our access token and access secret to Tweepy's user authentication handler
auth.set_access_token(access_token, access_token_secret)

# Creating a twitter API wrapper using tweepy
# Details here http://docs.tweepy.org/en/v3.5.0/api.html
api = tweepy.API(auth)
# if you use proxy, just add it to the API: proxy='https://your_proxy.server:port'

# Error handling
# Switching to application authentication
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

# Setting up new api wrapper, using authentication only
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# Error handling
if (not api):
    print("Problem Connecting to API")
searchQuery = 'Trumps'
#Maximum number of tweets we want to collect
maxTweets = 3000

#The twitter Search API allows up to 100 tweets per query
tweetsPerQry = 100
tweetCount = 0
csvFile = open('result.csv', 'a')

#Use csv writer
csvWriter = csv.writer(csvFile)
# Open a text file to save the tweets to

    # Tell the Cursor method that we want to use the Search API (api.search)
    # Also tell Cursor our query, and the maximum number of tweets to return
for tweet in tweepy.Cursor(api.search, q=searchQuery,lang='en').items():
        # Write a row to the CSV file. I use encode UTF-8
        csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
csvFile.close()

# Display how many tweets we have collected
print("Downloaded {0} tweets".format(tweetCount))