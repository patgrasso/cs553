#!/usr/bin/python3

import nltk
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

input = csv.reader(open("serious.csv", encoding="latin1"))
headers = next(input)

# read in the text from the csv
text = []
seriouses = []

for line in input:
    text.append(line[8])
    seriouses.append(line[-1])

# Partition the data into the 3 sets
test_size = int(len(text) * 0.1) # 10%
validate_size = int(len(text) * 0.1) # 10%

test_set = text[:test_size]
validate_set = text[test_size:test_size + validate_size]
train_set = text[test_size + validate_size:]

test_serious_set = text[:test_size]
validate_serious_set = text[test_size:test_size + validate_size]
train_serious_set = text[test_size + validate_size:]

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

from collections import defaultdict
cluster_dict = defaultdict(list)

for i in range(len(clusters)):
    cluster_dict[clusters[i]].append((
        train_set[i],
        train_serious_set[i]
    ))


for key in cluster_dict:
    print("Cluster {}".format(key))
    print("Size:", len(cluster_dict[key]))
    print("# serious:", sum([doc[1] == "Y" for doc in cluster_dict[key]]))
