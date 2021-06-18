import pandas as pd
import re, string
import matplotlib.pyplot as plt
import pickle

import gensim
from tqdm import tqdm

tqdm.pandas()

WORDVEC_SIZE = 100

def decontracted(phrase):
    phrase = re.sub(r"n't", " not", phrase)
    phrase = re.sub(r"nt ", " not ", phrase)
    phrase = re.sub(r"'re", " are", phrase)
    phrase = re.sub(r"youre", "you are", phrase)
    phrase = re.sub(r"'s", " is", phrase)
    phrase = re.sub(r"'d", " would", phrase)
    phrase = re.sub(r"'ll", " will", phrase)
    phrase = re.sub(r"'ve", " have", phrase)
    phrase = re.sub(r"'m", " am", phrase)
    phrase = re.sub(r"\#", "", phrase)
    return phrase

def remove_mentions(text):
    return re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)

def preprocess(text):
    text = decontracted(text.lower())
    punc = string.punctuation
    #whitelist = ["not", "no"]
    text = remove_mentions(text)
    #text = text.translate(str.maketrans("","", string.punctuation))
    text = re.sub(r'#', '', text)
    text = re.sub('["“]', '', text)
    text = re.sub('([.,:-;!$?()])', r' \1 ', text)
    #text = re.sub("\. \. \.", " ... ", text)
    return text

def clear_unused_fields(data, unused_fields):
    for field in unused_fields:
        data = data.drop(field,1)
    return data

def create_word_embeddings(data):
    print("creating word embeddings..")
    train_sentences = list(data['text'].progress_apply(str.split).values)
    model = gensim.models.Word2Vec(sentences=train_sentences,
                              vector_size=WORDVEC_SIZE,
                              window=5,
                              sg=1,
                              workers= 32,
                              seed = 34)
    model.train(train_sentences, total_examples= len(data['text']), epochs=20)
    return model
    
if __name__ == '__main__':
    data = pd.read_csv('twitter-airline-sentiment/Tweets.csv')
    pd.set_option('display.max_colwidth', None)
    data = clear_unused_fields(data, ['negativereason', 'negativereason_confidence', 'airline', 'airline_sentiment_gold', 
                                      'name', 'negativereason_gold', 'retweet_count', 'tweet_coord', 'tweet_created',
                                      'tweet_location','user_timezone', 'airline_sentiment_confidence', 'tweet_id'])
    data.text = data.text.apply(preprocess)
    word_embeddings_model = create_word_embeddings(data)
    word_embeddings_model.wv.save('vectors.kv')
    pickle.dump(word_embeddings_model, open("w2v_embeddings.pickle", 'wb'))
    