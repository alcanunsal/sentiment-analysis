import os, csv, re, string
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import preprocessor as p
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding, SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical
from nltk.stem import PorterStemmer, LancasterStemmer

api = KaggleApi()
api.authenticate()
#nltk.download('punkt')
#api.dataset_download_files('thoughtvector/customer-support-on-twitter')
#api.dataset_download_files('crowdflower/twitter-airline-sentiment')


def preprocess(row):
    txt = row["text"]
    if txt == "":
        return txt
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.NUMBER)
    txt = p.clean(txt)
    txt = txt.lower()
    #txt = ''.join([porter.stem(word) for word in txt.split()])
    #txt = re.sub(r'\d+', '', txt)
    #txt = txt.translate(str.maketrans("","", string.punctuation))
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
            f.close()
            return df
    except IOError:
        print("Download the dataset files first.")
        
def convert_sentiment(data):
    y = data['airline_sentiment']
    y = np.array(list(map(lambda x: 2 if x=="positive" else (1 if x=="negative" else 0), y)))
    return y

def tokenize(x_train, x_test):
    
    return x_train, x_test

def extract_features(x_train, x_test):
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(x_train)
    
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 100
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
    
    embeddings_dict = dict()
    glove_file = open('glove/glove.twitter.27B.100d.txt', encoding='utf8')
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dict[word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = np.zeros((vocab_size, 100))
    
    for word,index in tokenizer.word_index.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
    return x_train, x_test, embedding_matrix, vocab_size, maxlen
            
    
    

if __name__=='__main__':
    data = read_data("twitter-airline-sentiment/Tweets.csv")
    #sns.countplot(x='airline_sentiment', data=data)
    #plt.show()
    y = convert_sentiment(data)
    x = np.array(data["text"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    x_train, x_test, embedding_matrix, vocab_size, maxlen = extract_features(x_train, x_test)
    model = Sequential()
    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(128))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    
#sources:
#https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908 : preprocessing
#https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/ : initial lstm implementation
#https://nlp.stanford.edu/projects/glove/ : word embeddings