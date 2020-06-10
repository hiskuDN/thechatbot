# importing required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn import svm

# defining directories for datasets
entities_dir = './Data/entities_2.csv'
entities_model_dir = './Data/entities_model.pickle'

print("Training Entities")

entities_file = pd.read_csv(entities_dir)
entities_file.head()

# Reduce dataset side
entities_file = entities_file.sample(frac=1)
entities_file = entities_file[:50000]

training, testing = train_test_split(entities_file, test_size=0.33, random_state=55)

# Checking for and removing nan values
training = training.dropna(axis=0)
testing = testing.dropna(axis=0)

# Test and train mapped to X and y
# Training Data
train_x = training['Description']
train_y = training['Language']

# Testing Data
test_x = testing['Description']
test_y = testing['Language']

# Bag of word vectorization
vectorizer = CountVectorizer()
train_x_vector = vectorizer.fit_transform(train_x)
test_x_vector = vectorizer.transform(test_x)

"""## Classification
#### Linear SVM
"""

clf_svm = svm.SVC(kernel='linear')
clf_svm = Pipeline([('vect', vectorizer), ('svm', clf_svm)])
clf_svm.fit(train_x, train_y)

test_df = pd.DataFrame(testing)
test_df.head(15)

clf_svm.score(test_x, test_y)
clf_svm.predict(['I want an Angular library'])

# Saving model
with open(entities_model_dir, 'wb') as f:
    pickle.dump(clf_svm, f, pickle.HIGHEST_PROTOCOL)

# Load model from pickle for testing
f = open(entities_model_dir, 'rb')
pickle_model = pickle.load(f)

pickle_model.predict(['I want a React Library'])