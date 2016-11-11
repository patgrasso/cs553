#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk
import csv
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans


input = csv.reader(open("serious.csv", encoding="latin1"))
headers = next(input)

# read in the text from the csv
text = []
seriouses = []
tagged = []

for line in input:
    text.append(line[8])
    seriouses.append(line)
    tagged.append((line[8], line[-1]))

# Partition the data into the 3 sets
test_size = int(len(text) * 0.1) # 10%
validate_size = int(len(text) * 0.1) # 10%

test_set = text[:test_size]
validate_set = text[test_size:test_size + validate_size]
train_set = text[test_size + validate_size:]

test_serious_set = seriouses[:test_size]
validate_serious_set = seriouses[test_size:test_size + validate_size]
train_serious_set = seriouses[test_size + validate_size:]



# calculate the idf and tf_idf
from nltk.text import TextCollection
train_collection = TextCollection(train_set)
print("IDF of 'drug': ", train_collection.idf('drug'))
print("TF_IDF for first document: ", train_collection.tf_idf('drug', train_set[0]))

# get the TF IDF for all documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)

print("tfidf matrix shape:")
print(tfidf_matrix.shape)

import re
symptoms = train_set
m = map(lambda x: re.sub(r"[.,;]", "", x).split(' '), symptoms)
words = [word for doc in m for word in doc]
histogram = nltk.FreqDist(words)


common_words = sorted(histogram.items(), key=lambda x: x[1], reverse=True)[:30]
def extractor(text):
    features = {}
    for word in common_words:
        if word[0] in text:
            features['contains({})'.format(word)] = True
        else:
            features['contains({})'.format(word)] = False
    return features

featuresets = [(extractor(word), is_serious) for (word, is_serious) in tagged]

test_feature_set = featuresets[:test_size]
validate_feature_set = featuresets[test_size:test_size + validate_size]
train_feature_set = featuresets[test_size + validate_size:]

classifier = nltk.DecisionTreeClassifier.train(train_feature_set)
accuracy = nltk.classify.accuracy(classifier, test_feature_set)

print("accuracy", accuracy)
print(classifier.pseudocode(depth=4))
