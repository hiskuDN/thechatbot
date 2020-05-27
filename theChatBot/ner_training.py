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
from spacy.util import minibatch, compounding

gpu = spacy.prefer_gpu()
print('GPU:', gpu)

selected = pd.read_csv("enter_path_to_selected_2.csv")
selected = selected.dropna(axis=0)
selected.head()

# Strip punctuation from the description
import string

def stripPunctuation(txt):
  t = "".join([c for c in txt if c not in string.punctuation])
  return t


selected_train = selected[:50000]

train_s = []
# len(entities_s[1])
for index, row in selected_train.iterrows():
  train_s.append((stripPunctuation(str.strip(row['Description'])), {"entities": [(0, len(row['Description']), row['Name'])]}))

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
optimizer = nlp.begin_training()

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
output_dir = 'enter_the_output_file_path_here'
if output_dir is not None:
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)