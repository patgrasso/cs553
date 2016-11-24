#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk
from nltk.corpus import brown
import csv
import math


news = brown.sents(categories=['news'])[:3]
mystery = brown.sents(categories=['mystery'])[:3]
scifi = brown.sents(categories=['science_fiction'])[:3]

all = news + mystery + scifi
flattened = [val for sublist in all for val in sublist]
words = set(flattened)

spreadsheet = []
for doc in all:
    row = []
    for word in words:
        if word in doc:
            row.append(1)
        else:
            row.append(0)
    spreadsheet.append(row)

writer = csv.writer(open("./matrix.csv", 'w'))
writer.writerows(spreadsheet)



