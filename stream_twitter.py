import tweepy
from tweepy import StreamListener
from credentials import *
import sqlite3
from threading import Lock, Timer
import time
import json
from following import following
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

conn = sqlite3.connect('db/twitter_data.db', isolation_level=None, check_same_thread=False)
c = conn.cursor()


def create_table():
    try:

        # http://www.sqlite.org/pragma.html#pragma_journal_mode
        # for us - it allows concurrent write and reads
        c.execute("PRAGMA journal_mode=wal")
        c.execute("PRAGMA wal_checkpoint=TRUNCATE")
        # c.execute("PRAGMA journal_mode=PERSIST")

        # changed unix to INTEGER (it is integer, sqlite can use up to 8-byte long integers)
        c.execute(
            "CREATE TABLE IF NOT EXISTS tweets_table (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER, "
            "user_id INTEGER, name TEXT, user TEXT, tweet TEXT, status_id INTEGER, sentiment TEXT, source TEXT, "
            "verified TEXT, profile_pic TEXT, created_at DATE)")
        # key-value table for random stuff
        c.execute("CREATE TABLE IF NOT EXISTS misc(key TEXT PRIMARY KEY, value TEXT)")
        # id on index, both as DESC (as you are sorting in DESC order)
        c.execute("CREATE INDEX id_timestamp ON tweets_table (id DESC, timestamp DESC)")
        # out full-text search table, i choosed creating data from external (content) table - sentiment
        # instead of directly inserting to that table, as we are saving more data than just text
        # https://sqlite.org/fts5.html - 4.4.2
        c.execute(
            "CREATE VIRTUAL TABLE tweets_fts USING fts5(tweet, content=sentiment, content_rowid=id, "
            "prefix=1, prefix=2, prefix=3)")
        # that trigger will automatically update out table when row is inserted
        # (requires additional triggers on update and delete)
        c.execute("""
            CREATE TRIGGER tweet_insert AFTER INSERT ON tweets_table BEGIN
                INSERT INTO tweets_fts(rowid, tweet) VALUES (new.id, new.tweet);
            END
        """)
    except Exception as er:
        print(str(er))


create_table()

# create lock
lock = Lock()


class listener(StreamListener):
    data = []
    lock = None

    def __init__(self, lock):

        # create lock
        self.lock = lock

        # init timer for database save
        self.save_in_database()

        # call __inint__ of super class
        super().__init__()

    def save_in_database(self):

        # set a timer (1 second)
        Timer(1, self.save_in_database).start()

        # with lock, if there's data, save in transaction using one bulk query
        with self.lock:
            if len(self.data):
                # c.execute('BEGIN TRANSACTION')
                conn.commit()
                try:
                    c.executemany(
                        "INSERT INTO tweets_table (timestamp, user_id, name, user, tweet, status_id, sentiment, "
                        "source, verified, profile_pic, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", self.data)
                except:
                    pass
                # c.execute('COMMIT')
                conn.rollback()

                self.data = []

    def on_data(self, data):
        try:
            # print('data')
            data = json.loads(data)
            # print(data)
            # there are records like that:
            # {'limit': {'track': 14667, 'timestamp_ms': '1520216832822'}}
            if 'truncated' not in data:
                # print(data)
                return True
            if data['truncated']:
                tweet = data['extended_tweet']['full_text']
                #print(text_tweet)
            else:
                tweet = data['text']
                #print(text_tweet)

            user_id = data['user']['id']
            status_id = data['id']
            name = data['user']['name']
            user = data['user']['screen_name']
            source = data['source'].split('>')[1].split('<')[0]  # replace('Twitter for', '')
            created_at = data['created_at']
            # following = data['user']['following']
            eng = GoogleTranslator().translate(tweet)
            sentiment = str(analyzer.polarity_scores(eng))
            timestamp = data['timestamp_ms']
            verified = data['user']['verified']
            profile_pic = data['user']['profile_image_url_https']

            print(name, ' --- ', user, ' --- ', created_at)

            # append to data list (to be saved every 1 second)
            with self.lock:
                self.data.append(
                    (timestamp, user_id, name, user, tweet, status_id, sentiment, source, verified, profile_pic, created_at))

        except KeyError as e:
            print(str(e))
        return True

    def on_error(self, status):
        print(status)
        if status == '420':
            return False


while True:

    try:
        print(twitter_app)
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        twitterStream = tweepy.Stream(auth, listener(lock))
        twitterStream.filter(follow=following)
        # twitterStream.filter(track=['ə', 'ü', 'ç', 'ş', 'ö', 'a', 'e', 'i', 'u', 'o'])
    except Exception as e:
        print(str(e))
        time.sleep(5)
