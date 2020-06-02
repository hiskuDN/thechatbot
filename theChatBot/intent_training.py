# importing required libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# defining directories
intents_dir = './drive/My Drive/intents_mod.json'
intent_model_dir = './drive/My Drive/intents_model.pickle'

json_file = open(intents_dir)
intents = json.load(json_file)

intents_structured = []
X = []
y = []

for intent in intents['intents']:
  for pattern in intent['patterns']:
    y.append(intent['tag'])
    X.append(pattern)
    intents_structured.append([pattern, intent['tag']])

# Bag of word vectorization
vectorizer = CountVectorizer()
x_vector = vectorizer.fit_transform(X)

# Classification
from sklearn import svm

clf_svm = svm.SVC(kernel='linear')

clf_svm.fit(x_vector, y)

clf_svm.predict(vectorizer.transform(["Can you give me a nav bar library"]))

#Saving model
import pickle

with open(intent_model_dir, 'wb') as f:
  pickle.dump(clf_svm, f, pickle.HIGHEST_PROTOCOL)
