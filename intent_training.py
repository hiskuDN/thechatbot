# importing required libraries
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pickle
from sklearn import svm

# defining directories
intents_dir = './Data/intents_mod.json'
intent_model_dir = './Data/intents_model.pickle'

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

# classification
clf_svm = svm.SVC(kernel='linear')

clf_svm = Pipeline([('vect', vectorizer), ('svm', clf_svm)])

clf_svm.fit(X, y)

clf_svm.predict(["How famous is reat native"])

# saving model
with open(intent_model_dir, 'wb') as f:
    pickle.dump(clf_svm, f, pickle.HIGHEST_PROTOCOL)

# Testing saved model
f = open(intent_model_dir, 'rb')
pickle_model = pickle.load(f)

print(pickle_model.predict(['I want a React Library']))
