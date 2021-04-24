import sys, tweepy, pickle
'''authentication function'''
def twitter_auth():
    try: 
        consumer_key =''
        consumer_secret= ''
        access_token=''
        access_secret=''
    except KeyError:
        sys.stderr.write('TWITEER_* environment variable not set\n')
        sys.exit(1)
    auth= tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth
        
def get_twitter_client():
        auth= twitter_auth()
        client= tweepy.API(auth, wait_on_rate_limit=True)
        return client
    
if __name__ == '__main__':
        user=input("Enter username: ")
        client = get_twitter_client()
        tweets= tweepy.Cursor(client.user_timeline, screen_name=user).items()
       
            
        user_tweets=[status.text for status in tweets]
        with open('user_tweet', 'wb') as fp:
            pickle.dump(user_tweets, fp)    
        with open ('user_tweet', 'rb') as fp:
            user_tweets = pickle.load(fp)
        print(user_tweets)
            