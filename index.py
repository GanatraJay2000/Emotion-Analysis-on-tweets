from dotenv import load_dotenv
load_dotenv()
import os

import tweepy
from textblob import TextBlob

import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

# Set up API keys and access tokens
access_token = os.environ.get("OAUTH_TOKEN")
access_token_secret = os.environ.get("OAUTH_TOKEN_SECRET")
consumer_key = os.environ.get("CONSUMER_KEY")
consumer_secret = os.environ.get("CONSUMER_SECRET")

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define our search criteria
query = "Mumbai Indians vs Chennai Super Kings"
num_tweets = 100
positive_tweets = 0
negative_tweets = 0
neutral_tweets = 0

# Collect tweets that match our search criteria
tweets = tweepy.Cursor(api.search_tweets, q=query, lang='en').items(num_tweets)

def preprocess_tweet(tweet):
    # Remove URLs and any unnecessary characters
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#', "", tweet)
    tweet = re.sub(r'\d+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = tweet.strip()

    # Convert text to lowercase and split into tokens
    tokens = tweet.lower().split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into a string
    cleaned_tweet = " ".join(tokens)

    return cleaned_tweet

# Perform sentiment analysis on each tweet using TextBlob
for tweet in tweets:
    tweet_text = preprocess_tweet(tweet.text)
    blob = TextBlob(tweet_text)
    polarity, subjectivity = blob.sentiment
    print("Polarity:", polarity)  # polarity ranges from -1 (negative) to 1 (positive)
    print("Subjectivity:", subjectivity)  # subjectivity ranges from 0 (objective) to 1 (subjective)
    print("------------------------")
    print()
    print()
    print()
    print()
    if polarity > 0:
        positive_tweets +=1
    elif polarity < 0:
        negative_tweets +=1
    else:
        neutral_tweets +=1


labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive_tweets, negative_tweets, neutral_tweets]
plt.bar(labels, sizes)

# add title and labels
plt.title('Sentiment Analysis Results')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')

# show the plot
plt.show()