# Bot GUI goes here
import pandas as pd
import os
import ast
import json
import re
import numpy as np
import random
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import gensim
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

path = './Data/'
entities_pickle = './Data/entities_model.pickle'

# load trained data
try:
    desc_json_path = path + 'Desc_Word2Vec.json'

    with open(desc_json_path) as file:
        reader = json.load(file)

        langs = []
        descriptions = []
        desc_tokens = []
        ids = []
        names = []
        repourls = []
        desc_lengths = []
        desc_vectors = []
        average_pooling = []
        for row in reader:
            langs.append(row['Lang'])
            descriptions.append(row['Description'])
            desc_tokens.append(row['Desc_Tokens'].split())
            ids.append(row['ID'])
            names.append(row['Name'])
            repourls.append(row['RepoUrl'])
            desc_lengths.append(row['Desc_Length'])
            desc_vectors.append(row['Desc_Vectors'])
            average_pooling.append(row['Average_Pooling'])

        data_tokens = pd.DataFrame({'Lang': langs,
                                    'Description': descriptions,
                                    'Desc_Tokens': desc_tokens,
                                    'ID': ids,
                                    'Name': names,
                                    'RepoUrl': repourls,
                                    'Desc_Length': desc_lengths,
                                    'Desc_Vectors': desc_vectors,
                                    'Average_Pooling': average_pooling})
except:
    pass


def clean_up_sentence(sentence):
    stop_words = stopwords.words("english")

    # Remove non english words
    sentence = [re.sub('[^a-z(JavaScript)(Python)(Java)(PHP)(C++)]', ' ', x.lower()) for x in sentence]
    # Tokenization
    sentence_tokens = [nltk.word_tokenize(t) for t in sentence]
    # Removing Stop Words
    sentence_tokens = [[t for t in tokens if (t not in stop_words) and (3 < len(t.strip()) < 15)]
                       for tokens in sentence_tokens]

    sentence_tokens = pd.Series(sentence_tokens)
    return sentence_tokens


def matchRepoToInput(data_by_language, model, msg_tokens):
    cosines = []
    try:
        # Get vectors and average pooling
        question_vectors = []
        for token in msg_tokens:
            try:
                vector = model[token]
                question_vectors.append(vector)
            except:
                continue
        question_ap = list(pd.DataFrame(question_vectors[0]).mean())

        # Calculate cosine similarity
        for t in data_by_language['Average_Pooling']:
            if t is not None and len(t) == len(question_ap):
                val = cosine_similarity([question_ap], [t])
                cosines.append(val[0][0])
            else:
                cosines.append(0)
    except:
        pass

    # If not in the topic trained
    if len(cosines) == 0:
        not_understood = "Apology, I do not understand. Can you rephrase?"
        return not_understood, 999

    else:
        # Sort similarity
        index_s = []
        score_s = []
        for i in range(len(cosines)):
            x = cosines[i]
            if x >= 0.9:
                index_s.append(i)
                score_s.append(cosines[i])

        reply_indexes = pd.DataFrame({'index': index_s, 'score': score_s})
        reply_indexes = reply_indexes.sort_values(by="score", ascending=False)

        # Find Top Questions and Score
        r_index = int(reply_indexes['index'].iloc[0])
        r_score = float(reply_indexes['score'].iloc[0])

        # reply = str(data_by_language.iloc[:,0][r_index])
        reply = str("My suggestion: " + data_by_language['Description'].iloc[r_index] + ". Repo name: " +
                    data_by_language['Name'].iloc[r_index] + ". Repo URL: " + data_by_language['RepoUrl'].iloc[r_index])

        return reply, r_score


def chatbot_response(msg):
    # add code to determine lang here, for now hardcode as python
    f = open(entities_pickle, 'rb')
    pickle_model = pickle.load(f)
    vec = CountVectorizer()

    # TODO Pass the predicted language output to the generic lookup model
    predicted_language = pickle_model.predict(vec.transform([msg]))
    language = "Python"
    data_by_language = data_tokens[data_tokens['Lang'] == predicted_language]
    data_by_language = pd.DataFrame({'Description': list(data_by_language['Description']),
                                     'Desc_Tokens': list(data_by_language['Desc_Tokens']),
                                     'ID': list(data_by_language['ID']),
                                     'Name': list(data_by_language['Name']),
                                     'RepoUrl': list(data_by_language['RepoUrl']),
                                     'Lang': list(data_by_language['Lang']),
                                     'Desc_Vectors': list(data_by_language['Desc_Vectors']),
                                     'Average_Pooling': list(data_by_language['Average_Pooling'])
                                     })
    word2vec_pickle_path = path + 'desc_word2vec_' + language + '.bin'
    model = gensim.models.KeyedVectors.load(word2vec_pickle_path)
    msg_tokens = clean_up_sentence(pd.Series(msg))
    reply, score = matchRepoToInput(data_by_language, model, msg_tokens)
    return reply


# Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
