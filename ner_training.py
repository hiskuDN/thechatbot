# importing required libraries
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import json
import pandas as pd
import numpy as np

# Import required modules
import spacy
import random
import thinc_gpu_ops
from spacy.util import minibatch, compounding
from pathlib import Path

print(thinc_gpu_ops.AVAILABLE)
gpu = spacy.prefer_gpu()
print('GPU:', gpu)

selected = pd.read_csv("./Data/selected_2.csv")
selected = selected.dropna(axis=0)
selected.head()

# Strip punctuation from the description
import string


def stripPunctuation(txt):
    t = "".join([c for c in txt if c not in string.punctuation])
    return t


selected_train = selected[:50000]
# selected_train = selected

train_s = []
# len(entities_s[1])
for index, row in selected_train.iterrows():
    # train_s.append((stripPunctuation(str.strip(row['Description'])), {"entities": [(0, len(row['Description']), row['Name'])]}))
    processedStr = stripPunctuation(str.strip(row['Name']))
    if len(processedStr) > 0:
        tempSentence = processedStr + " is a library"
        tempSentence2 = "i want to know about " + processedStr
        tempSentence3 = "what is good about " + processedStr + "?"
        tempSentence4 = "the library " + processedStr + " is pretty good"
        tempSentence5 = "how popular is " + processedStr + " in your opinion?"
        tempSentence6 = processedStr + " is a good library"
        tempSentence7 = "is " + processedStr + " famous?"
        tempSentence8 = "how many stars does " + processedStr + " have?"
        tempSentence9 = "why is " + processedStr + " a good library"
        tempSentence10 = "is " + processedStr + " popular?"
        tempSentence11 = "how famous is " + processedStr + "?"
        tempSentence12 = "do a lot of people use " + processedStr + "?"
        tempSentence13 = "is " + processedStr + " adopted by a lot of users?"
        tempSentence14 = "famous library " + processedStr
        tempSentence15 = processedStr + " popularity"
        tempSentence16 = "popularity of " + processedStr
        tempSentence17 = "ratings of " + processedStr
        tempSentence18 = "how good is " + processedStr + "?"
        tempSentence19 = "i want to know how good " + processedStr + "is"
        tempSentence20 = "what are the dependencies of " + processedStr + "?"
        tempSentence21 = "give me a list of all " + processedStr + " dependencies"
        tempSentence22 = "what libraries does " + processedStr + " depends on?"
        tempSentence23 = "who wrote " + processedStr + "?"
        tempSentence24 = "who is the author of " + processedStr + "?"
        tempSentence25 = "who was " + processedStr + " written by?"
        tempSentence26 = "author of " + processedStr
        tempSentence27 = processedStr + " author"
        tempSentence28 = "who was " + processedStr + " created by?"
        tempSentence29 = "who created " + processedStr + "?"
        tempSentence30 = "who is " + processedStr + "'s writer?"
        tempSentence31 = "writer of " + processedStr
        tempSentence32 = "i want to know who the writer of " + processedStr + " is"
        tempSentence33 = "i wanted to know who the writer of " + processedStr + " was"
        tempSentence34 = "who is the writer of " + processedStr + "?"
        tempSentence35 = "give me the author of " + processedStr
        tempSentence36 = "i am looking for the writer of " + processedStr
        tempSentence37 = "are there other libraries by the author of " + processedStr + "?"
        tempSentence38 = "other libraries like " + processedStr
        tempSentence39 = "i am looking for other libraries like " + processedStr
        tempSentence40 = "other libraries by the writer of " + processedStr
        tempSentence41 = "what other libraries did the writer of " + processedStr + " write?"
        train_s.append((tempSentence, {"entities": [(0, len(processedStr), "libName")]}))
        train_s.append((tempSentence2, {"entities": [(21, 21 + len(processedStr), "libName")]}))
        train_s.append((tempSentence3, {"entities": [(19, 19 + len(processedStr), "libName")]}))
        train_s.append((tempSentence4, {"entities": [(12, 12 + len(processedStr), "libName")]}))
        train_s.append((tempSentence5, {"entities": [(15, 15 + len(processedStr), "libName")]}))
        train_s.append((tempSentence6, {"entities": [(0, len(processedStr), "libName")]}))
        train_s.append((tempSentence7, {"entities": [(3, 3 + len(processedStr), "libName")]}))
        train_s.append((tempSentence8, {"entities": [(20, 20 + len(processedStr), "libName")]}))
        train_s.append((tempSentence9, {"entities": [(7, 7 + len(processedStr), "libName")]}))
        train_s.append((tempSentence10, {"entities": [(3, 3 + len(processedStr), "libName")]}))
        train_s.append((tempSentence11, {"entities": [(14, 14 + len(processedStr), "libName")]}))
        train_s.append((tempSentence12, {"entities": [(23, 23 + len(processedStr), "libName")]}))
        train_s.append((tempSentence13, {"entities": [(3, 3 + len(processedStr), "libName")]}))
        train_s.append((tempSentence14, {"entities": [(15, 15 + len(processedStr), "libName")]}))
        train_s.append((tempSentence15, {"entities": [(0, len(processedStr), "libName")]}))
        train_s.append((tempSentence16, {"entities": [(14, 14 + len(processedStr), "libName")]}))
        train_s.append((tempSentence17, {"entities": [(11, 11 + len(processedStr), "libName")]}))
        train_s.append((tempSentence18, {"entities": [(12, 12 + len(processedStr), "libName")]}))
        train_s.append((tempSentence19, {"entities": [(24, 24 + len(processedStr), "libName")]}))
        train_s.append((tempSentence20, {"entities": [(29, 29 + len(processedStr), "libName")]}))
        train_s.append((tempSentence21, {"entities": [(22, 22 + len(processedStr), "libName")]}))
        train_s.append((tempSentence22, {"entities": [(20, 20 + len(processedStr), "libName")]}))
        train_s.append((tempSentence23, {"entities": [(10, 10 + len(processedStr), "libName")]}))
        train_s.append((tempSentence24, {"entities": [(21, 21 + len(processedStr), "libName")]}))
        train_s.append((tempSentence25, {"entities": [(8, 8 + len(processedStr), "libName")]}))
        train_s.append((tempSentence26, {"entities": [(10, 10 + len(processedStr), "libName")]}))
        train_s.append((tempSentence27, {"entities": [(0, len(processedStr), "libName")]}))
        train_s.append((tempSentence28, {"entities": [(8, 8 + len(processedStr), "libName")]}))
        train_s.append((tempSentence29, {"entities": [(12, 12 + len(processedStr), "libName")]}))
        train_s.append((tempSentence30, {"entities": [(7, 7 + len(processedStr), "libName")]}))
        train_s.append((tempSentence31, {"entities": [(10, 10 + len(processedStr), "libName")]}))
        train_s.append((tempSentence32, {"entities": [(33, 33 + len(processedStr), "libName")]}))
        train_s.append((tempSentence33, {"entities": [(35, 35 + len(processedStr), "libName")]}))
        train_s.append((tempSentence34, {"entities": [(21, 21 + len(processedStr), "libName")]}))
        train_s.append((tempSentence35, {"entities": [(22, 22 + len(processedStr), "libName")]}))
        train_s.append((tempSentence36, {"entities": [(31, 31 + len(processedStr), "libName")]}))
        train_s.append((tempSentence37, {"entities": [(43, 43 + len(processedStr), "libName")]}))
        train_s.append((tempSentence38, {"entities": [(21, 21 + len(processedStr), "libName")]}))
        train_s.append((tempSentence39, {"entities": [(38, 38 + len(processedStr), "libName")]}))
        train_s.append((tempSentence40, {"entities": [(33, 33 + len(processedStr), "libName")]}))
        train_s.append((tempSentence41, {"entities": [(39, 39 + len(processedStr), "libName")]}))

print(train_s[:41])

train_s = np.array(train_s)

# create the built-in pipeline components and add them to the pipeline
nlp = spacy.blank("en")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)

# add new entitiy labels
labels = []
for t in train_s:
    ner.add_label(t[1]['entities'][0][2])

# training
optimizer = nlp.begin_training(device=0)

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
n_iter = 10

with nlp.disable_pipes(*other_pipes):  # only train NER
    for itn in range(n_iter):
        random.shuffle(train_s)
        losses = {}
        batches = minibatch(train_s, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            # Updating the weights
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print('Losses', losses)
        nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
        print('Losses', losses)

# save model to output directory
output_dir = './Data/'
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
