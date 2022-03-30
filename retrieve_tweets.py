from kafka import KafkaProducer
import tweepy
import datetime
import time
import json

client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAGVSaAEAAAAAWThCSC%2BRmpY8paAGsu5fav%2FzqxE%3DVlhxBhzyQnoncRIfQLOUFBPrmwaiNKoeqPczYG97Me0GxwS9BB')
producer = KafkaProducer()
producer.flush()

query = 'covid'
start_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=15)

while True:
    end_time = datetime.datetime.utcnow()
    print('Fifteen seconds break!')
    time.sleep(15)
    tweets = client.search_recent_tweets(query=query,
                                        tweet_fields=['context_annotations', 'created_at', 'lang'], 
                                        max_results=100, 
                                        start_time=start_time,
                                        end_time=end_time)
    start_time = end_time
    for i,tweet in enumerate(tweets.data):
        if tweet.lang == 'en':
            tweet = json.dumps(tweet.text).encode('utf-8')
            producer.send('trump', tweet)
        print(f'Tweet {i} was successfully sent to kafka')
    

