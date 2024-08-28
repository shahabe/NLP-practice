import praw
import pandas as pd
import datetime as dt

reddit = praw.Reddit(client_id = '<The client_id or the personal use script>',
                     client_secret = '<The secret variable in your reddit app>',
                     username='<your user name or the developers variable>',
                     password='<your password>',
                     user_agent='<This can be any phrase or the name of your api>')

topic_name = "Covid19"
subreddit = reddit.subreddit(topic_name)

topics_dict = { "title":[], \
                "score":[], \
                "id":[], "url":[], \
                "comms_num": [], \
                "created": [], \
                "post_comment":[], \
                "upvotes":[]}
for submission in subreddit.hot(limit=100):
    topics_dict["title"].append(submission.title)
    topics_dict["score"].append(submission.score)
    topics_dict["id"].append(submission.id)
    topics_dict["url"].append(submission.url)
    topics_dict["comms_num"].append(submission.num_comments)
    topics_dict["created"].append(submission.created)
    topics_dict["post_comment"].append(submission.selftext)
    topics_dict["upvotes"].append(submission.ups)
    
    topics_data = pd.DataFrame(topics_dict)


def get_date(created):
    return dt.datetime.fromtimestamp(created)

_timestamp = topics_data["created"].apply(get_date)
topics_data = topics_data.assign(timestamp = _timestamp)


topics_data.to_csv(f'{topic_name}_topics.csv') 