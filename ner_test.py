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

# Strip punctuation from the description
import string

def stripPunctuation(txt):
  t = "".join([c for c in txt if c not in string.punctuation])
  return t


output_dir = './Data/'
if output_dir is not None:
    output_dir = Path(output_dir)
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)
    doc = nlp("How often is hackernewsapi updated?")
    doc2 = nlp("Can you tell me more about hackernewsapi?")
    doc3 = nlp("How many people use hackernewsapi?")
    doc4 = nlp("How popular is hackernewsapi?")
    doc5 = nlp("How many stars does hackernewsapi has?")
    print(doc)
    for ent in doc.ents:
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    print(doc2)
    for ent in doc2.ents:
        print("Entities", [(ent.text, ent.label_) for ent in doc2.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc2])
    print(doc3)
    for ent in doc3.ents:
        print("Entities", [(ent.text, ent.label_) for ent in doc3.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc3])
    print(doc4)
    for ent in doc4.ents:
        print("Entities", [(ent.text, ent.label_) for ent in doc4.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc4])
    print(doc5)
    for ent in doc5.ents:
        print("Entities", [(ent.text, ent.label_) for ent in doc5.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc5])