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


# Define the emotion labels
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
count_per_emotion = {'anger':0, 'disgust':0, 'fear':0, 'joy':0, 'sadness':0, 'surprise':0}


# Define our search criteria
query = "premier league"
num_tweets = 1000
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
for indTweet, tweet in enumerate(tweets):
    tweet_text = preprocess_tweet(tweet.text)
    blob = TextBlob(tweet_text)
    emotions = blob.sentiment
    emotion_values = [(emotions.polarity + 1) / 2, emotions.subjectivity]
    # print("Polarity:", polarity)  # polarity ranges from -1 (negative) to 1 (positive)
    # print("Subjectivity:", subjectivity)  # subjectivity ranges from 0 (objective) to 1 (subjective)
    emotion = max(emotion_values)
    e_label = emotion_labels[emotion_values.index(emotion)]
    count_per_emotion[e_label] += 1
    print()
    print(indTweet+1, ". ", tweet_text)
    print("Emotion: ",e_label,": ", emotion)
    print("------------------------")
    print()
    print()
    print()
    print()

labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
sizes = [count_per_emotion['anger'], count_per_emotion['disgust'], count_per_emotion['fear'], count_per_emotion['joy'], count_per_emotion['sadness'], count_per_emotion['surprise']]
plt.bar(labels, sizes)

# add title and labels
plt.title('Sentiment Analysis Results')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')

# show the plot
plt.show()