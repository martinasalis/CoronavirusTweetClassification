## Imports
from gensim.parsing.preprocessing import remove_stopwords
from gensim import models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules, Adam, SGD, RMSprop
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from enum import Enum

import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import string
import gensim
import plotly.graph_objects as go
import tensorflow as tf
import gensim.downloader
import warnings
warnings.filterwarnings('ignore')


class Model(Enum):
  GOOGLE = 1
  TWITTER = 2
  DATASET = 3
  EMBEDDING = 4


def remove_url(text):
  url = re.compile(r'https?://\S+|www\.\S+')
  return url.sub(r'',text)


def remove_html(text):
  html=re.compile(r'<.*?>')
  return html.sub(r'',text)


def remove_emoji(text):
  emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r'', text)


def decontraction(text):
  text = re.sub(r"won\'t", " will not", text)
  text = re.sub(r"won\'t've", " will not have", text)
  text = re.sub(r"can\'t", " can not", text)
  text = re.sub(r"don\'t", " do not", text)

  text = re.sub(r"can\'t've", " can not have", text)
  text = re.sub(r"ma\'am", " madam", text)
  text = re.sub(r"let\'s", " let us", text)
  text = re.sub(r"ain\'t", " am not", text)
  text = re.sub(r"shan\'t", " shall not", text)
  text = re.sub(r"sha\n't", " shall not", text)
  text = re.sub(r"o\'clock", " of the clock", text)
  text = re.sub(r"y\'all", " you all", text)

  text = re.sub(r"n\'t", " not", text)
  text = re.sub(r"n\'t've", " not have", text)
  text = re.sub(r"\'re", " are", text)
  text = re.sub(r"\'s", " is", text)
  text = re.sub(r"\'d", " would", text)
  text = re.sub(r"\'d've", " would have", text)
  text = re.sub(r"\'ll", " will", text)
  text = re.sub(r"\'ll've", " will have", text)
  text = re.sub(r"\'t", " not", text)
  text = re.sub(r"\'ve", " have", text)
  text = re.sub(r"\'m", " am", text)
  text = re.sub(r"\'re", " are", text)
  return text


def seperate_alphanumeric(text):
    words = text
    words = re.findall(r"[^\W\d_]+|\d+", words)
    return " ".join(words)


def cont_rep_char(text):
    tchr = text.group(0)

    if len(tchr) > 1:
        return tchr[0:2]


def unique_char(rep, text):
    substitute = re.sub(r'(\w)\1+', rep, text)
    return substitute


def char(text):
    substitute = re.sub(r'[^a-zA-Z]', ' ', text)
    return substitute


def conv_sentiment(tweet_sentiment):
  if tweet_sentiment == "Extremely Positive":
    return "Positive"
  elif tweet_sentiment == "Extremely Negative":
    return "Negative"
  else:
    return tweet_sentiment


def conv_sentiment_to_int(tweet_sentiment):
  if tweet_sentiment == "Positive":
    return 0
  elif tweet_sentiment == "Negative":
    return 1
  else:
    return 2


def sent_vectorizer(sent, model):
  sent_vec = []
  numw = 0
  for w in sent:
    try:
      if numw == 0:
        sent_vec = model[w]
      else:
        sent_vec = np.add(sent_vec, model[w])
      numw+=1
    except:
      pass

  return np.asarray(sent_vec)/numw


def main():
  ## Flags
  create_dataset = False
  model_type = Model.GOOGLE
  do_baseline = False
  do_network = True

  ## Create dataset
  if(create_dataset):
    # Load train data
    train = pd.read_csv('./dataset/Corona_NLP_train.csv', encoding="latin1")

    # Load test data
    test = pd.read_csv('./dataset/Corona_NLP_test.csv', encoding="latin1")

    # Merge train and test data
    dataset = pd.concat([train, test])

    ## Clean text
    # Apply functions on OriginalTweet of Train set
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : remove_url(x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : remove_emoji(x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : decontraction(x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : seperate_alphanumeric(x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : unique_char(cont_rep_char,x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : char(x))
    dataset['OriginalTweet'] = dataset['OriginalTweet'].apply(lambda x : x.lower())

    # Split dataset

    X = dataset['OriginalTweet']
    Y = dataset['Sentiment']

    # Split data into train , test
    X_train , X_test , Y_train , Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # Split data into train , validation
    X_train , X_val , Y_train , Y_val = train_test_split(X_train, Y_train, test_size=0.11, random_state=0)

    num_label_train_index, num_label_train_values = np.unique(Y_train, return_counts=True)
    num_label_test_index, num_label_test_values = np.unique(Y_test, return_counts=True)
    num_label_val_index, num_label_val_values = np.unique(Y_val, return_counts=True)

    # Save splitted dataset

    np.save('./dataset/X_train', X_train, allow_pickle=True, fix_imports=True)
    np.save('./dataset/X_test', X_test, allow_pickle=True, fix_imports=True)
    np.save('./dataset/X_val', X_val, allow_pickle=True, fix_imports=True)

    np.save('./dataset/y_train', Y_train, allow_pickle=True, fix_imports=True)
    np.save('./dataset/y_test', Y_test, allow_pickle=True, fix_imports=True)
    np.save('./dataset/y_val', Y_val, allow_pickle=True, fix_imports=True)

  else:
    #Load splitted dataset

    X_train = pd.Series(np.load('./dataset/X_train.npy', allow_pickle=True))
    X_test = pd.Series(np.load('./dataset/X_test.npy', allow_pickle=True))
    X_val = pd.Series(np.load('./dataset/X_val.npy', allow_pickle=True))

    Y_train = pd.Series(np.load('./dataset/y_train.npy', allow_pickle=True))
    Y_test = pd.Series(np.load('./dataset/y_test.npy', allow_pickle=True))
    Y_val = pd.Series(np.load('./dataset/y_val.npy', allow_pickle=True))

    num_label_train_index, num_label_train_values = np.unique(Y_train, return_counts=True)
    num_label_test_index, num_label_test_values = np.unique(Y_test, return_counts=True)
    num_label_val_index, num_label_val_values = np.unique(Y_val, return_counts=True)

  # Convert five labels to three labels
  Y_train = Y_train.apply(lambda x: conv_sentiment(x))
  Y_test = Y_test.apply(lambda x: conv_sentiment(x))
  Y_val = Y_val.apply(lambda x: conv_sentiment(x))

  num_label_train_index, num_label_train_values = np.unique(Y_train, return_counts=True)
  num_label_test_index, num_label_test_values = np.unique(Y_test, return_counts=True)
  num_label_val_index, num_label_val_values = np.unique(Y_val, return_counts=True)

  # Convert string labels to integer labels
  Y_train = Y_train.apply(lambda x: conv_sentiment_to_int(x))
  Y_test = Y_test.apply(lambda x: conv_sentiment_to_int(x))
  Y_val = Y_val.apply(lambda x: conv_sentiment_to_int(x))

  ## Word2Vec model
  #Upload pre-trained model
  if(model_type == Model.GOOGLE):
    print("GOOGLE vocab")
    # Upload Gooogle News model
    wv_from_bin = models.KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin', binary=True)

    # Length of words model
    maxlen = 300

    vocab = wv_from_bin.vocab
    vectors = wv_from_bin.vectors

  elif(model_type == Model.TWITTER):
    print("TWITTER vocab")
    # Upload Twitter model
    wv_from_bin = gensim.downloader.load('glove-twitter-100')

    # Length of words model
    maxlen = 100

    vocab = wv_from_bin.vocab
    vectors = wv_from_bin.vectors

  elif(model_type == Model.DATASET):
    print("DATASET vocab")
    # Train model
    wv_from_bin = models.Word2Vec(sentences=list(X_train.apply(lambda x: x.split(' '))), min_count=1)

    # Length of words model
    maxlen = 100

    vocab = wv_from_bin.wv.vocab
    vectors = wv_from_bin.wv.vectors

  else:
    # Length of words model
    maxlen = 100

  ## Baseline
  if(do_baseline):
    # Vectorized tweets
    x_train = []
    for sentence in list(X_train):
        x_train.append(sent_vectorizer(sentence, wv_from_bin))

    x_test = []
    for sentence in list(X_test):
        x_test.append(sent_vectorizer(sentence, wv_from_bin))

    x_val = []
    for sentence in list(X_val):
        x_val.append(sent_vectorizer(sentence, wv_from_bin))

    x_train = pad_sequences(np.array(x_train), maxlen=maxlen)
    x_test = pad_sequences(np.array(x_test), maxlen=maxlen)
    x_val = pad_sequences(np.array(x_val), maxlen=maxlen)

    y_test = Y_test
    y_train = Y_train
    y_val = Y_val

    # Multi-layer Perceptron
    clf= MLPClassifier()
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    print("Multi-layer Perceptron")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Multi-layer Perceptron with Cross Validation
    parameters = {}
    clf = GridSearchCV(MLPClassifier(), parameters, cv=5).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Multi-layer Perceptron with Cross Validation")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(x_train, y_train).predict(x_test)
    print("Gaussian Naive Bayes")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Gaussian Naive Bayes with Cross Validation
    parameters = {}
    clf = GridSearchCV(GaussianNB(), parameters, cv=5).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Gaussian Naive Bayes with Cross Validation")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Logistic Regression
    clf = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Logistic Regression")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Logistic Regression with Cross Validation
    parameters = {}
    clf = GridSearchCV(LogisticRegression(random_state=0, solver='liblinear'), parameters, cv=5).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Logistic Regression with Cross Validation")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # SVM
    clf = SVC(kernel='linear', max_iter=100, decision_function_shape='ovo')
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    print("SVM")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # SVM with Cross Validation
    parameters = {}
    clf = GridSearchCV(SVC(kernel='linear', max_iter=100, decision_function_shape='ovo'), parameters, cv=5).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("SVM with Cross Validation")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Random Forest
    clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=250)
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    print("Random Forest")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

    # Random Forest with Cross Validation
    parameters = {}
    clf = GridSearchCV(RandomForestClassifier(max_depth=50, random_state=0, n_estimators=250), parameters, cv=5).fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("Random Forest with Cross Validation")
    print("Number of mislabeled points out of a total %d points : %d" % (x_test.shape[0], (y_test != y_pred).sum()))
    print(clf.score(x_test, y_test))

  ## New Network
  if(do_network):
    y_test = Y_test
    y_train = Y_train
    y_val = Y_val

    # Convert y to categorical
    y_train = to_categorical(y_train, 3)
    y_test  = to_categorical(y_test, 3)
    y_val  = to_categorical(y_val, 3)

    max_features = 30000
    tokenizer = Tokenizer(num_words=max_features) # num_words=max_features,
    tokenizer.fit_on_texts(list(X_train))

    list_tokenized_train = tokenizer.texts_to_sequences(X_train)
    list_tokenized_val = tokenizer.texts_to_sequences(X_val)
    list_tokenized_test = tokenizer.texts_to_sequences(X_test)

    # convert tokenized texts into same padding size
    embed_size = 128
    X_train_final = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_val_final = pad_sequences(list_tokenized_val, maxlen=maxlen)
    X_test_final = pad_sequences(list_tokenized_test, maxlen=maxlen)

    size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
    print(size_of_vocabulary)

    embeddings_index = {}
    for word, vector in zip(vocab, vectors):
      coefs = np.asarray(vector, dtype='float32')
      embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, maxlen))
    for word, i in tokenizer.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

    model=Sequential()

    #embedding layer
    model.add(Embedding(len(tokenizer.word_index) + 1, maxlen, weights=[embedding_matrix], input_length=maxlen, trainable=False))
    #model.add(Embedding(max_features, embed_size))

    #lstm layer
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)))

    #Global Maxpooling
    model.add(GlobalMaxPooling1D())

    #Dense Layer
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='sigmoid'))

    # Learning rate decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.5)

    #Add loss function, metrics, optimizer
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=["acc"])

    #Print summary of model
    print(model.summary())

    history = model.fit(X_train_final, y_train, epochs=10, validation_data=(X_val_final, y_val), batch_size=64)

    _,test_acc = model.evaluate(X_test_final, y_test)
    print(test_acc)


if __name__ == "__main__":
  main()
