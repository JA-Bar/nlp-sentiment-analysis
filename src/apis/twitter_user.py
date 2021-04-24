import os

import tweepy
import pickle


def twitter_auth():
    consumer_key = os.environ.get('TWITTER_CONSUMER_KEY')
    consumer_secret = os.environ.get('TWITTER_CONSUMER_SECRET')
    access_token = os.environ.get('TWITTER_ACCESS_TOKEN')
    access_secret = os.environ.get('TWITTER_ACCESS_SECRET')

    if not all([consumer_key, consumer_secret, access_token, access_secret]):
        raise AttributeError('TWITEER_* environment variable not set\n')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth


def get_twitter_client():
    auth = twitter_auth()
    client = tweepy.API(auth, wait_on_rate_limit=True)
    return client


def tweets_from_user(user=None):
    if user is None:
        user = input("Enter username: ")

    client = get_twitter_client()
    tweets = tweepy.Cursor(client.user_timeline, screen_name=user).items()

    user_tweets = [status.text for status in tweets]
    with open('user_tweet', 'wb') as fp:
        pickle.dump(user_tweets, fp)
    with open('user_tweet', 'rb') as fp:
        user_tweets = pickle.load(fp)

    return user_tweets


if __name__ == '__main__':
    print(tweets_from_user())

