#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk
import csv
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

input = csv.reader(open("serious.csv", encoding="latin1"))
headers = next(input)

# read in the text from the csv
text = []
seriouses = []

samples = []

for line in input:
    samples += [(line[8], line[-1])]
    text.append(line[8])
    seriouses.append(line)

# Partition the data into the 3 sets
test_size = int(len(text) * 0.7) # 10%
validate_size = int(len(text) * 0.2) # 10%

test_set = text[:test_size]
validate_set = text[test_size:test_size + validate_size]
train_set = text[test_size + validate_size:]

test_serious_set = seriouses[:test_size]
validate_serious_set = seriouses[test_size:test_size + validate_size]
train_serious_set = seriouses[test_size + validate_size:]

test_samples = samples[:test_size]
validate_samples = samples[test_size:test_size + validate_size]
train_samples = samples[test_size + validate_size:]

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)

symptoms = map(lambda x: x[0], train_samples)
m = map(lambda x: re.sub(r"[.,;]", "", x).split(' '), symptoms)
words = [word for doc in m for word in doc]
histogram = nltk.FreqDist(words)

print(sorted(histogram.items(), key=lambda x:x[1], reverse=True)[:10])

exit(0)

classifier = nltk.DecisionTreeClassifier.train(train_samples)
print(classifier.pseudocode(depth=4))



