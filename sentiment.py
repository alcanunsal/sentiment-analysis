import os, csv, re, string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Embedding, Bidirectional, Dropout
from keras.layers.recurrent import LSTM
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
import time, pickle

from gensim.models import KeyedVectors, Word2Vec
from tqdm import tqdm

tqdm.pandas()

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

def load_embeddings():
    reloaded_word_vectors = KeyedVectors.load('vectors.kv')
    return reloaded_word_vectors

def create_embedding_matrix(x_train, x_test):
    x_train = list(x_train)
    x_test = list(x_test)
    tokenizer = Tokenizer(num_words=len(vocab))
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    
    embeddings_dict = dict()
    for word in vocab:
        embeddings_dict[word] = list(embeddings[word])    
    
    embedding_matrix = np.zeros((vocab_size, 100))
    
    for word,index in tokenizer.word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    return x_train, x_test, embedding_matrix, vocab_size, maxlen

def read_tweets():
    data = pd.read_csv('twitter-airline-sentiment/Tweets.csv')
    pd.set_option('display.max_colwidth', None)
    data = clear_unused_fields(data, ['negativereason', 'negativereason_confidence', 'airline', 'airline_sentiment_gold', 
                                      'name', 'negativereason_gold', 'retweet_count', 'tweet_coord', 'tweet_created',
                                      'tweet_location','user_timezone', 'airline_sentiment_confidence', 'tweet_id'])
    data.text = data.text.apply(preprocess)
    return data

def labels_to_categorical(y_train,y_test):
    lb = LabelBinarizer()
    y_train = list(y_train)
    y_test = list(y_test)
    y_train_categorical = lb.fit_transform(y_train)
    y_test_categorical = lb.fit_transform(y_test)
    return y_train_categorical, y_test_categorical
 
if __name__=='__main__':
    embeddings = load_embeddings()
    vocab = list(embeddings.key_to_index.keys())
    data = read_tweets()

    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['airline_sentiment'], test_size=0.2)
    x_train, x_test, embedding_matrix, vocab_size, maxlen = create_embedding_matrix(x_train, x_test)

    y_train_categorical, y_test_categorical = labels_to_categorical(y_train, y_test)
    
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64,dropout=0.2,recurrent_dropout=0.3)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='binary_crossentropy',optimizer="adam",metrics=['accuracy'])
    history = model.fit(x_train, y_train_categorical, batch_size=32, epochs=20, verbose=1, validation_split=0.2)
    score = model.evaluate(x_test, y_test_categorical, verbose=1)
    