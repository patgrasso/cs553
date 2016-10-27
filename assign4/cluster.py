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

for line in input:
    text.append(line[8])
    seriouses.append(line)

text = text[:int(len(text)/6)]

# Partition the data into the 3 sets
test_size = int(len(text) * 0.1) # 10%
validate_size = int(len(text) * 0.1) # 10%

test_set = text[:test_size]
validate_set = text[test_size:test_size + validate_size]
train_set = text[test_size + validate_size:]

test_serious_set = seriouses[:test_size]
validate_serious_set = seriouses[test_size:test_size + validate_size]
train_serious_set = seriouses[test_size + validate_size:]

# get the TF IDF for all documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)

print("tfidf matrix shape:")
print(tfidf_matrix.shape)

# run k means with 5 clusters
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

# print the first 10 documents with the cluster they belong to
#for i in range(10):
#    print('cluster: {:d}, text: {}'.format(clusters[i], train_set[i]))

from collections import defaultdict
cluster_dict = defaultdict(list)

for i in range(len(clusters)):
    cluster_dict[clusters[i]].append((
        train_set[i],
        train_serious_set[i]
    ))


# Print how many documents belong in each cluster and how many of them were serious incidents
for key in cluster_dict:
    print("Cluster {}".format(key))
    print("Size:", len(cluster_dict[key]))
    print("# serious:", sum([doc[1][-1] == "Y" for doc in cluster_dict[key]]))


# Average of each variable
avg_hist = { header: {} for header in headers }

print("Segment Profile:")

hists = []
avg_hist = [ {} for header in headers ]
for key in cluster_dict:
    # for each cluster
    cluster_hist = [ {} for header in headers ]
    for doc in cluster_dict[key]:
        for var in range(len(headers)):
            if doc[1][var] not in cluster_hist[var]:
                cluster_hist[var][doc[1][var]] = 0
            if doc[1][var] not in avg_hist[var]:
                avg_hist[var][doc[1][var]] = 0
            cluster_hist[var][doc[1][var]] += 1
            avg_hist[var][doc[1][var]] += 1
    hists += [cluster_hist]

for hist_i in range(len(hists)):
    hist = hists[hist_i]
    print("cluster", hist_i)
    dists = []
    for var in range(len(hist)):
        if headers[var] == "SYMPTOM_TEXT":
            continue
        #print(" ", headers[var])
        dist = 0
        for key in hist[var]:
            pop_avg = avg_hist[var][key] * 1.0 / sum(avg_hist[var][k] for k in avg_hist[var])
            cls_avg = hist[var][key] * 1.0 / sum(hist[var][k] for k in hist[var])
            dist += (pop_avg - cls_avg)**2
            #print("   ", key, ":", pop_avg - cls_avg)
        dist /= 1.0 * len(hist[var])
        dists += [(math.sqrt(dist), headers[var])]
    for x in list(reversed(sorted(dists)))[:6]:
        print(x[1], x[0])




