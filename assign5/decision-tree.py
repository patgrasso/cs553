#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk, csv, math, re, pandas, scipy, string
import numpy as np
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from tficf import TficfVectorizer
from nltk.stem.porter import PorterStemmer

from printtree import tree_to_code

warnings.simplefilter("ignore", np.VisibleDeprecationWarning)

NUM_TERMS_TO_PRINT = 10
TRAIN_RATIO = 0.8

def report(results, n_top=3):
    """
    Report the top n hyperparameter configurations for this set
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results["mean_test_score"][candidate],
                results["std_test_score"][candidate]
            ))
            print("Parameters: {0}".format(results["params"][candidate]))
            print()

def tokenize(dataframe, column_label=""):
    """
    Filter stop words and punctuation
    """
    series = dataframe
    if type(dataframe) != pandas.Series:
        series = dataframe[column_label]

    stop_words = nltk.corpus.stopwords.words("english") +\
                 [sym for sym in string.punctuation]

    remove_stops = lambda sent: [token for token in
                                 nltk.word_tokenize(sent.lower())
                                 if token not in stop_words]
    stemmer = PorterStemmer()
    stem_tokens = lambda tokens: [stemmer.stem(token) for token in tokens]

    return dataframe[column_label].apply(remove_stops).apply(stem_tokens)


samples = pandas.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(frac=TRAIN_RATIO)
test_samples = samples.drop(train_samples.index)
#test_yes = test_samples[test_samples["SERIOUS"] == "Y"]
#test_no = test_samples[test_samples["SERIOUS"] == "N"].sample(test_yes.shape[0])
#test_samples = pandas.concat((test_no, test_yes))

print("Train : {} ({:.2%})".format(
    train_samples.shape[0],
    train_samples.shape[0] / samples.shape[0]))
print("Test  : {} ({:.2%})".format(
    test_samples.shape[0],
    test_samples.shape[0] / samples.shape[0]))

vectorizers = {
    "tfidf": TfidfVectorizer(max_df=0.99, max_features=1000,
                             min_df=0.01, stop_words="english",
                             use_idf=True, ngram_range=(1,3)),
    "count": CountVectorizer(max_df=0.99, max_features=1000,
                             min_df=0.01, stop_words="english",
                             ngram_range=(1,3)),
    "tficf": TficfVectorizer(max_df=0.99, max_features=1000,
                             min_df=0.01, stop_words="english",
                             use_idf=True, ngram_range=(1,3))
}


# Pick vectorization method
vectorizer = vectorizers["tfidf"]


# Filter stop words and punctuation
print("Removing stop words/stemming")
train_samples["SYMPTOM_TEXT"] = tokenize(train_samples, "SYMPTOM_TEXT")\
                                .apply(lambda x: ' '.join(x))
test_samples["SYMPTOM_TEXT"] = tokenize(test_samples, "SYMPTOM_TEXT")\
                               .apply(lambda x: ' '.join(x))

print("Vectorizing with {}".format(vectorizer.__class__.__name__))

word_matrix = vectorizer.fit_transform(
    train_samples["SYMPTOM_TEXT"],
    train_samples["SERIOUS"])

try:
    test_matrix = vectorizer.transform(
        test_samples["SYMPTOM_TEXT"],
        test_samples["SERIOUS"])
except:
    test_matrix = vectorizer.transform(
        test_samples["SYMPTOM_TEXT"])

if isinstance(word_matrix, scipy.sparse.csr_matrix):
    word_matrix = word_matrix.toarray()
    test_matrix = test_matrix.toarray()

print("Vectorization complete. Classifying...")


# Set up decision tree
clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=6)
baseline = DummyClassifier(strategy="prior")

clf.fit(word_matrix, train_samples["SERIOUS"])
baseline.fit(word_matrix, train_samples["SERIOUS"])


# Print tree
#tree_to_code(clf.best_estimator_, vectorizer.get_feature_names())
export_graphviz(clf, out_file = "tree.dot",
                feature_names = vectorizer.get_feature_names(),
                class_names = ["N", "Y"],
                filled = True,
                impurity = False,
                proportion = True)


# Get feature importances
feature_importances = sorted(zip(
    clf.feature_importances_,
    vectorizer.get_feature_names()
), reverse=True)


# Print the top N important features
print()
print("Printing the top %d significant features" % NUM_TERMS_TO_PRINT)
for importance, feature in feature_importances[:NUM_TERMS_TO_PRINT]:
    feature = (feature[:27] + "...") if len(feature) > 30 else feature
    print("{:30s}: {}".format(feature, importance))


# Print classifier's accuracy on the test set
clf_conf = confusion_matrix(
    test_samples["SERIOUS"],
    clf.predict(test_matrix))
base_conf = confusion_matrix(
    test_samples["SERIOUS"],
    baseline.predict(test_matrix))

print()
print("tree accuracy  :", clf.score(test_matrix, test_samples["SERIOUS"]))
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

