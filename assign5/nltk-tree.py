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
from sklearn.metrics import confusion_matrix, accuracy_score

from printtree import tree_to_code

warnings.simplefilter("ignore", np.VisibleDeprecationWarning)

TRAIN_RATIO = 0.8

def to_feature_dict(matrix, feature_names, serious_col=None):
    lst = []
    for row in range(len(matrix)):
        if serious_col is not None:
            lst += [(
                { feature_names[i]: bool(matrix[row][i])
                  for i in range(len(matrix[row])) },
                serious_col.iloc[row]
            )]
        else:
            lst += [{ feature_names[i]: bool(matrix[row][i])
                      for i in range(len(matrix[row])) }]
    return lst

samples = pandas.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(frac=TRAIN_RATIO)
test_samples = samples.drop(train_samples.index)

print("Train : {} ({:.2%})".format(
    train_samples.shape[0],
    train_samples.shape[0] / samples.shape[0]))
print("Test  : {} ({:.2%})".format(
    test_samples.shape[0],
    test_samples.shape[0] / samples.shape[0]))

vectorizers = {
    "tfidf": TfidfVectorizer(max_df=0.8, max_features=200000,
                             min_df=0.2, stop_words='english',
                             use_idf=True, ngram_range=(1,3)),
    "count": CountVectorizer(max_df=0.99, max_features=1000,
                             min_df=0.01, stop_words="english",
                             ngram_range=(1,3))
}

# Pick vectorization method
vectorizer = vectorizers["count"]

print("Vectorizing with {}".format(vectorizer.__class__.__name__))
word_matrix = vectorizer.fit_transform(train_samples["SYMPTOM_TEXT"]).toarray()
test_matrix = vectorizer.transform(test_samples["SYMPTOM_TEXT"]).toarray()

what_nltk_wants = to_feature_dict(
    word_matrix,
    vectorizer.get_feature_names(),
    train_samples["SERIOUS"])

print("Vectorization complete. Classifying...")
clf = nltk.DecisionTreeClassifier.train(what_nltk_wants)
baseline = DummyClassifier(strategy="prior").fit(
    word_matrix, train_samples["SERIOUS"])

what_nltk_wants = to_feature_dict(
    test_matrix,
    vectorizer.get_feature_names())

print(clf.pseudocode(depth=6))

results = pandas.concat((
    pandas.Series(map(clf.classify, what_nltk_wants), name = "predict"),
    test_samples["SERIOUS"].reset_index(drop=True)
), axis = 1)


# Print classifier's accuracy on the test set
clf_conf = confusion_matrix(results["SERIOUS"], results["predict"])
base_conf = confusion_matrix(
    test_samples["SERIOUS"],
    baseline.predict(test_matrix))

print()
print("tree accuracy  :", accuracy_score(
    results["SERIOUS"],
    results["predict"]))
print("tree recall    :", clf_conf[1,1] / (clf_conf[1,1] + clf_conf[1,0]))
print("tree precision :", clf_conf[1,1] / (clf_conf[1,1] + clf_conf[0,1]))
print("prior accuracy :", baseline.score(test_matrix, test_samples["SERIOUS"]))
print("prior recall   :", base_conf[1,1] / (base_conf[1,1] + base_conf[1,0]))
print("prior precision:", base_conf[1,1] / (base_conf[1,1] + base_conf[0,1]+1))
print()
print("~ confusion ~")
print("reference:")
print(np.array([["TN", "FP"], ["FN", "TP"]]))
print()
print("confusion [%s]:" % clf.__class__.__name__)
print(clf_conf)
print()
print("confusion [%s]:" % baseline.__class__.__name__)
print(base_conf)

