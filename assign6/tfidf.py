#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import csv
import math

stops = stopwords.words('english')

# Get the texts wer are going to use and make useful variables from them
news = brown.sents(categories=['news'])[:3]
mystery = brown.sents(categories=['mystery'])[:3]
scifi = brown.sents(categories=['science_fiction'])[:3]

all = news + mystery + scifi
flattened = [val for sublist in all for val in sublist]
words = set(flattened)
df = {}
for word in words:
    df[word] = sum([1 if word in doc else 0 for doc in flattened])

# write documents to files

news_file = open('news.txt', 'w+')
for doc in news:
    news_file.write(' '.join(doc))
    news_file.write("\nUnique Words:\n")
    for word in doc:
        if flattened.count(word) == 1:
            news_file.write(word)
            news_file.write(", ")
    news_file.write("\n\n")


mystery_file = open('mystery.txt', 'w+')
for doc in mystery:
    mystery_file.write(' '.join(doc))
    mystery_file.write("\nUnique Words:\n")
    for word in doc:
        if flattened.count(word) == 1:
            mystery_file.write(word)
            mystery_file.write(", ")
    mystery_file.write("\n\n")

scifi_file = open('scifi.txt', 'w+')
for doc in scifi:
    scifi_file.write(' '.join(doc))
    scifi_file.write("\nUnique Words:\n")
    for word in doc:
        if flattened.count(word) == 1:
            scifi_file.write(word)
            scifi_file.write(", ")
    scifi_file.write("\n\n")

# create the spreadsheet of tf-idfs
spreadsheet = []
termfrequencies = []
for doc in all:
    row = []
    rowfrequencies = []
    for word in words:
        tf = doc.count(word)
        row.append(tf * math.log(len(flattened) / df[word]))
        rowfrequencies.append(tf)

    spreadsheet.append(row)
    termfrequencies.append(rowfrequencies)

# write it to a file
writer = csv.writer(open("./tfidf.csv", 'w'))
writer.writerows(spreadsheet)

writer = csv.writer(open("./tf.csv", 'w'))
writer.writerows(termfrequencies)

# calculate distances
distances = []
for row in spreadsheet:
    result = []
    for otherrow in spreadsheet:
        distance = 0
        for i in range(len(row)):
            distance += (row[i] - otherrow[i]) ** 2
        result.append(distance)
    distances.append(result)

# We can see from printing all the distances that documents from the same category are closer
print(distances)

