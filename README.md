# sentiment-analysis


this is the repository for my graduation project: 
sentiment analysis on twitter for brands and corporations. twitter mentions for airline companies are used as the training data.

the model uses word embeddings and a bidirectional LSTM neural network to predict the sentiment (positive, negative, neutral) of a tweet/customer feedback.

further improvements will include using part-of-speech tags and dependency information of words in addition to word embeddings.

### current test accuracy score: 81%

dataset sources:

- https://www.kaggle.com/crowdflower/twitter-airline-sentiment


You need to download the dataset before running the code. 

How to run:

1. create a new conda environment and activate:

> conda create -n <env-name> python=3.6
  
> conda activate <env-name>
  
2. install requirements
  
> python3 -m pip install -r requirements.txt
  
3. create word embeddings
  
> python3 word_embeddings.py
  
4. create and evaluate model
  
> python3 sentiment.py
  
