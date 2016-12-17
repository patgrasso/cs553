
#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk, csv, math, re, pandas
import numpy as np
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from printtree import tree_to_code

warnings.simplefilter("ignore", np.VisibleDeprecationWarning)

def to_feature_dict(matrix, serious_col, feature_names):
    print(matrix)
    lst = []
    for row in range(len(matrix)):
        lst += [(
            { feature_names[i]: matrix[row][i]
              for i in range(len(matrix[row])) },
            serious_col.iloc[row]
        )]
    return lst

samples = pandas.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(frac=0.7)
test_samples = samples.drop(train_samples.index)

print("Training on {} samples".format(train_samples.shape[0]))

vectorizers = {
    "tfidf": TfidfVectorizer(max_df=0.8, max_features=200000,
                             min_df=0.2, stop_words='english',
                             use_idf=True, ngram_range=(1,3)),
    "count": CountVectorizer(stop_words="english", ngram_range=(1,3),
                             max_features=1000)
}


# Pick vectorization method
vectorizer = vectorizers["tfidf"]
word_matrix = vectorizer.fit_transform(train_samples["SYMPTOM_TEXT"]).toarray()
test_matrix = vectorizer.transform(test_samples["SYMPTOM_TEXT"]).toarray()


# Set up decision tree
decision_tree = DecisionTreeClassifier()


what_nltk_wants = to_feature_dict(
        word_matrix,
        train_samples["SERIOUS"],
        vectorizer.get_feature_names())

#clf = nltk.DecisionTreeClassifier.train(what_nltk_wants)

#print("score:", clf.error(what_nltk_wants))

#print(clf.pseudocode(depth=4))
