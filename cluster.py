#!/usr/bin/python3

import nltk
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

input = csv.reader(open("serious.csv"))
headers = next(input)

# read in the text from the csv
text = [line[8] for line in input]


# Partition the data into the 3 sets
test_size = int(len(text) * 0.1) # 10%
validate_size = int(len(text) * 0.1) # 10%

test_set = text[:test_size]
validate_set = text[test_size:test_size + validate_size]
train_set = text[test_size + validate_size:]

# get the TF IDF for all documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)

print(tfidf_matrix.shape)

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# print the first 10 documents with the cluster they belong to
for i in range(10):
    print('cluster: {:d}, text: {}'.format(clusters[i], train_set[i]))

import pandas as pd

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace

print()
print()
