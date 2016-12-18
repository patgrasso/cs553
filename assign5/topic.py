#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import warnings

warnings.simplefilter("ignore", UserWarning)

import pandas as pd
import numpy as np
import gensim
import nltk
import string

samples = pd.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(100)
test_samples = samples.drop(train_samples.index)

# Filter stop words and punctuation
stop_words = nltk.corpus.stopwords.words("english") +\
             [sym for sym in string.punctuation]

remove_stops = lambda sent: [token for token in
                             nltk.word_tokenize(sent.lower())
                             if token not in stop_words]

symptoms = train_samples["SYMPTOM_TEXT"].apply(remove_stops)
train_samples["SYMPTOM_TEXT"] = symptoms

# Create LDA/LSI model for topic extraction
dictionary = gensim.corpora.Dictionary(train_samples["SYMPTOM_TEXT"])
corpus = [dictionary.doc2bow(text) for text in train_samples["SYMPTOM_TEXT"]]
tfidf = gensim.models.TfidfModel(corpus)
tcorpus = [tfidf[e] for e in corpus]

model = gensim.models.lsimodel.LsiModel(corpus = tcorpus,
                                        id2word = dictionary,
                                        num_topics = 15)

topic_words = [[w for w, _ in tups] for _, tups in
               model.show_topics(formatted = 0, num_words = 10)]

for topic in topic_words:
    print(topic)




