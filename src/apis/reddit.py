import os
import re

import praw
from dotenv import load_dotenv


load_dotenv()
USER_AGENT = 'pc:nlp.sentiment_analyzer.bot:v0.0.1 (by u/NLP_Project_BOT)'


def comments_from_user(username, limit=10):
    CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
    CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
    if not (CLIENT_ID and CLIENT_SECRET):
        raise AttributeError("Missing environment variables for reddit credentials.")

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )

    redditor = reddit.redditor(username)

    comments = redditor.comments
    comments_list = comments.new(limit=limit)

    pattern_url = re.compile(r'\[(.+)\]\(.+\)')  # markdown links
    pattern_subreddit = re.compile(r'r/\w+\b')  # subreddits
    pattern_user = re.compile(r'u/\w+\b')  # users

    processed_text = []
    for c in comments_list:
        text = c.body
        text = re.sub(pattern_url, lambda x: x.groups()[0], text)
        text = re.sub(pattern_subreddit, '_SUB_', text)
        text = re.sub(pattern_user, '_USR_', text)
        processed_text.append(text)

    return processed_text


if __name__ == '__main__':
    print(comments_from_user('johnc2001'))

