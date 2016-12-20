#!/usr/bin/python3

# To get this set up, you will need to install nltk, and nltk-data
# I installed the book dataset, but there are good instructions on the nltk site
import nltk
import csv


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

test_doc = "\n".join(test_set)
validate_doc = "\n".join(validate_set)
train_doc = "\n".join(train_set)

# Do some processing with NLTK

print('first training set and its tokens:')
print(train_set[0])
tokens = nltk.word_tokenize(train_set[0])
print(tokens)
print()

# prints the type of each token
tagged = nltk.pos_tag(tokens)
print('sample of token tagging')
print(tagged[0:6])
print()

# calculate the idf and tf_idf
from nltk.text import TextCollection
train_collection = TextCollection(train_set)
print("IDF of 'drug': ", train_collection.idf('drug'))
print("TF_IDF for first document: ", train_collection.tf_idf('drug', train_set[0]))

# Calculate word frequencies
from nltk.probability import FreqDist
# Get the frequencies of all words
fdist = FreqDist(nltk.word_tokenize(train_doc))
print()



print(fdist)
pairs = [(key, fdist[key]) for key in fdist]
print(pairs[:10])



# Do some processing with spaCy
# Commented this out since it takes forever to run and you would have to get spaCy working
# However it does look like spaCy is a good alternative library for natural language processing with python
# and many people are starting to switch to it recently
# import spacy
# nlp = spacy.load('en')
# train_doc = nlp(train_set[0])
# sentence = next(train_doc.sents)

