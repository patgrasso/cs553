#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk, csv, math, re, pandas
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier

samples = pandas.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(frac=0.5)
test_samples = samples.drop(train_samples.index)

print(train_samples.shape)

#tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
#                                 min_df=0.1, stop_words='english',
#                                 use_idf=True, ngram_range=(1,3))
#
#tfidf_matrix = tfidf_vectorizer.fit_transform(train_samples["SYMPTOM_TEXT"])
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1,3),
                             max_features=1000)
word_matrix = vectorizer.fit_transform(train_samples["SYMPTOM_TEXT"]).toarray()
test_matrix = vectorizer.fit_transform(test_samples["SYMPTOM_TEXT"]).toarray()
print(word_matrix.shape)
#features = tfidf_vectorizer.get_feature_names()
#features = count_vectorizer.

#symptoms = map(lambda x: x[0], train_samples)
#m = map(lambda x: re.sub(r"[.,;]", "", x).split(' '), symptoms)
#words = [word for doc in m for word in doc]
#histogram = nltk.FreqDist(words)

#print(sorted(histogram.items(), key=lambda x:x[1], reverse=True)[:10])

print("Training Decision Tree Classifier...")
#print(train_samples[0])
#classifier = nltk.DecisionTreeClassifier.train(train_samples)
#print(classifier.pseudocode(depth=4))
clf = DecisionTreeClassifier(max_depth=30)

clf.fit(word_matrix, train_samples["SERIOUS"])
print("score:", clf.score(test_matrix, test_samples["SERIOUS"]))



