#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

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

test_serious_set = seriouses[:test_size]
validate_serious_set = seriouses[test_size:test_size + validate_size]
train_serious_set = seriouses[test_size + validate_size:]

# get the TF IDF for all documents
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(train_set)

print(tfidf_matrix.shape)

# run k means with 5 clusters
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


# Print how many documents belong in each cluster and how many of them were serious incidents
for key in cluster_dict:
    print("Cluster {}".format(key))
    print("Size:", len(cluster_dict[key]))
    print("# serious:", sum([doc[1] == "Y" for doc in cluster_dict[key]]))


# do heirarchical clustering
# was unable to get this working in the end, possibly the sample size was too large

# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import cosine_similarity

# dist = 1 - cosine_similarity(tfidf_matrix)
# print('dist calculated')

# from scipy.cluster.hierarchy import ward, dendrogram

# linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
# print('warded')

# fig, ax = plt.subplots(figsize=(15, 20)) # set size
# ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

# plt.tick_params(\
#     axis= 'x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off')

# plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
# print('saved')

# plt.tight_layout() #show plot with tight layout

