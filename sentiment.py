import os, csv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import preprocessor as p

api = KaggleApi()
api.authenticate()
#api.dataset_download_files('thoughtvector/customer-support-on-twitter')
#api.dataset_download_files('crowdflower/twitter-airline-sentiment')


def preprocess(row):
    txt = row["text"]
    if txt == "":
        return txt
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER)
    txt = p.clean(txt)
    txt = txt.lower()
    return txt

def read_data(dataset_path):

    try:
        with open(dataset_path) as f:
            reader = list(csv.reader(f))
            indices=reader[0]
            data = reader[1:]
            #tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,airline_sentiment_gold,name,negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
            df = pd.DataFrame(data, columns=indices)
            del df["retweet_count"]
            del df["tweet_location"]
            del df["user_timezone"]
            del df["tweet_created"]
            del df["tweet_coord"]
            df["text"]=df.apply(preprocess, axis=1)
            return df
    except IOError:
        print("Download the dataset files first.")


if __name__=='__main__':
    data = read_data("twitter-airline-sentiment/Tweets.csv")

#570306133677760513,neutral,1.0,,,Virgin America,,cairdin,,0,@VirginAmerica What @dhepburn said.,,2015-02-24 11:35:52 -0800,,Eastern Time (US & Canada)