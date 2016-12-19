#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import warnings

warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import numpy as np
import gensim
import nltk
import string

from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

samples = pd.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

NUM_TOPICS = 100
TRAIN_RATIO = 0.8

# Partition the data into training/testing sets
train_samples = samples.sample(frac=TRAIN_RATIO)
test_samples = samples.drop(train_samples.index)

print("Train : {} ({:.2%})".format(
    train_samples.shape[0],
    train_samples.shape[0] / samples.shape[0]))
print("Test  : {} ({:.2%})".format(
    test_samples.shape[0],
    test_samples.shape[0] / samples.shape[0]))


# Filter stop words and punctuation
def tokenize(dataframe, column_label=""):
    series = dataframe
    if type(dataframe) != pd.Series:
        series = dataframe[column_label]

    stop_words = nltk.corpus.stopwords.words("english") +\
                 [sym for sym in string.punctuation]

    remove_stops = lambda sent: [token for token in
                                 nltk.word_tokenize(sent.lower())
                                 if token not in stop_words]
    stemmer = PorterStemmer()
    stem_tokens = lambda tokens: [stemmer.stem(token) for token in tokens]

    return dataframe[column_label].apply(remove_stops).apply(stem_tokens)


# Apply tokenization to training/testing sets
print("Removing stop words/stemming")
train_samples["SYMPTOM_TEXT"] = tokenize(train_samples, "SYMPTOM_TEXT")
test_samples["SYMPTOM_TEXT"] = tokenize(test_samples, "SYMPTOM_TEXT")


# Create LDA/LSI model for topic extraction
print("Creating tf-idf models")
dictionary = gensim.corpora.Dictionary(train_samples["SYMPTOM_TEXT"])
train_corpus = [dictionary.doc2bow(text)
                for text in train_samples["SYMPTOM_TEXT"]]
test_corpus = [dictionary.doc2bow(text)
               for text in test_samples["SYMPTOM_TEXT"]]
tfidf = gensim.models.TfidfModel(train_corpus)
train_tfidf_corpus = [tfidf[e] for e in train_corpus]
test_tfidf_corpus = [tfidf[e] for e in test_corpus]

print("Creating LSI model")
model = gensim.models.lsimodel.LsiModel(
    corpus = train_tfidf_corpus,
    id2word = dictionary,
    num_topics = NUM_TOPICS)

topic_words = [(index, [w for w, _ in tups]) for index, tups in
               model.show_topics(num_topics = 300,
                                 formatted = 0,
                                 num_words = 5)]

# Check out our dope topics
print("Topics:")
for topic in topic_words:
    print(topic)

train_X = pd.DataFrame([dict(model[sample])
                        for sample in train_tfidf_corpus],
                       columns = pd.RangeIndex(NUM_TOPICS)).fillna(0)
train_y = train_samples["SERIOUS"].reset_index(drop=True)

test_X = pd.DataFrame([dict(model[sample])
                       for sample in test_tfidf_corpus],
                       columns = pd.RangeIndex(NUM_TOPICS)).fillna(0)
test_y = test_samples["SERIOUS"].reset_index(drop=True)

clf = RandomForestClassifier()

print()
print("Training a %s on the topic probability matrix" % clf.__class__.__name__)
clf.fit(train_X, train_y)
baseline = DummyClassifier(strategy="prior").fit(train_X, train_y)


# Print classifier's accuracy on the test set
clf_conf = confusion_matrix(test_y, clf.predict(test_X))
base_conf = confusion_matrix(test_y, baseline.predict(test_X))

print()
print("topic accuracy :", clf.score(test_X, test_y))
print("topic recall   :", clf_conf[1,1] / (clf_conf[1,1] + clf_conf[1,0]))
print("topic precision:", clf_conf[1,1] / (clf_conf[1,1] + clf_conf[0,1]))
print("prior accuracy :", baseline.score(test_X, test_y))
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

