#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import csv
import math
import numpy as np

stops = stopwords.words('english')

N_SAMPLES = 3

# Get the texts we are going to use and make useful variables from them
news = brown.sents(categories=['news'])[:N_SAMPLES]
mystery = brown.sents(categories=['mystery'])[:N_SAMPLES]
scifi = brown.sents(categories=['science_fiction'])[:N_SAMPLES]

def lowerify(collection):
    return [[word.lower() for word in sent] for sent in collection]

def remove_stops(collection):
    return [list(filter(lambda word: word not in stops, sent)) for sent in collection]

news = remove_stops(lowerify(news))
mystery = remove_stops(lowerify(mystery))
scifi = remove_stops(lowerify(scifi))

all = news + mystery + scifi
flattened = [val for sublist in all for val in sublist]
words = sorted(set(flattened))
df = {}
for word in words:
    df[word] = sum([1 if word in doc else 0 for doc in all])

# write documents to files
def writeDocToFile(fd, collection):
    for doc in collection:
        fd.write("Document:     ")
        fd.write(" ".join(doc))
        fd.write("\nUnique Words: ")
        unique = filter(lambda word: df[word] == 1, doc)
        fd.write(", ".join(unique))
        fd.write("\n\n")

writeDocToFile(open("news.txt", "w+"), news)
writeDocToFile(open("mystery.txt", "w+"), mystery)
writeDocToFile(open("scifi.txt", "w+"), scifi)
open("stop_words.txt", "w+").write("\n".join(stops))
print("Topic Data [news]    : news.txt")
print("Topic Data [mystery] : news.txt")
print("Topic Data [scifi]   : news.txt")
print("Stop Words           : stop_words.txt")

# create the spreadsheet of tf-idfs
spreadsheet = []
termfrequencies = []
for doc in all:
    row = []
    rowfrequencies = []
    for word in words:
        tf = doc.count(word) / float(len(doc))
        row.append(tf * (1 + math.log(len(all) / float(df[word]))))
        rowfrequencies.append(tf)

    spreadsheet.append(row)
    termfrequencies.append(rowfrequencies)

# write it to a file
print("TFIDF Table          : tfidf.csv");
writer = csv.writer(open("./tfidf.csv", 'w'))
writer.writerows([words] + spreadsheet)

print("Term Frequency Table : tf.csv");
writer = csv.writer(open("./tf.csv", 'w'))
writer.writerows([words] + termfrequencies)

# calculate similarities
similarities = []
for row in spreadsheet:
    result = []
    for otherrow in spreadsheet:
        dot = 0
        magRow = 0
        magOtherRow = 0
        for i in range(len(row)):
            if row[i] != 0.0:
                dot += (row[i] * otherrow[i])
                magRow += row[i] ** 2
                magOtherRow += otherrow[i] ** 2
        magRow **= 0.5
        magOtherRow **= 0.5
        try: distance = dot / (magRow * magOtherRow)
        except: distance = 0.0
        result.append(distance)
    similarities.append(result)

# compare average similarities from different topics using cosine similarity
# exclude the diagonal (same documents will always be perfectly similar)
topics = []
for i in range(0, 3*N_SAMPLES, N_SAMPLES):
    row = []
    for j in range(0, 3*N_SAMPLES, N_SAMPLES):
        similar = 0
        df = 0
        for k in range(N_SAMPLES):
            for m in range(N_SAMPLES):
                if i + k != j + m:
                    similar += similarities[i + k][j + m] ** 2
                else:
                    df += 1
        row += [(similar / ((N_SAMPLES ** 2) - df))**0.5]
    topics += [row]


# We can see from printing all the similarities that documents from the same
# category are closer
np.set_printoptions(linewidth=float('inf'))
np.savetxt("doc_similarity.csv", similarities, fmt="%10.8f", delimiter=',')

print()
print("Document similarities: doc_similarity.csv")
print(np.array(similarities))

# This prints the normalized average of the similarities for each topic
# The diagonal indicates how closely related documents of the same topic are
# Other elements indicate how closely related the docs from one topic are to
# those of another
print()
print("Topic similarities   : topic_similarity.csv")
np.savetxt("topic_similarity.csv", topics, fmt="%10.8f", delimiter=',')
print(np.array(topics))


