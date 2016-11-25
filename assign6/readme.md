# Assignment 6 - "By hand" Clustering

Patrick Grasso and Nicholas Antonov worked together on this assignment.

## Dataset / topics
The data are from the nltk "brown" corpus. We took sentences from three topics -
`news`, `mystery`, and `scifi`. `N_SAMPLES` controls the number of sentences
used from each topic.

## Filtering
First, we lower-cased all of the words and removed the english stop-words
included with nltk (all of these words are included in `stop_words.txt`).
Because of this, the sentences listed in each topic file might not read well as
they are missing common english words.

## tf-idf
Next, we counted the occurrences of each word in each document. From that, we
could determine which words were unique to a document. Every document and the
respective unique words for each topic can be found in `news.txt`,
`mystery.txt`, and `scifi.txt`. Then, we found the tf-idf of these words for
each document and stored it in a matrix, `tfidf.csv`. The separate term
frequencies are in `tf.csv`.

## Cosine similarity
Using the tfidfs for each term and document, we computed the similarity between
each document and every other document using cosine similarity, which is
computed like so:
```
  dot(docA, docB) = sum(tfidf(term, docA) * tfidf(term, docB)) for term in docA
  ||docA|| = sqrt(sum(tfidf(term, docA)^2)) for term in docA
  ||docB|| = sqrt(sum(tfidf(term, docB)^2)) for term in docA
  similarity = dot(docA, docB) / ||docA|| * ||docB||
```
This produces the correlation matrix in `doc_similarity.csv`. Notice that the
diagonal is filled with ones, which makes sense because every document should be
perfectly similar to itself.

## Topic similarity
Lastly, we took the square-average of document similarities for each topic.
Basically, we averaged each `N_SAMPLES x N_SAMPLES` square in the similarity
matrix to see the relationship between each set of documents
(`topic_similarity.csv`).

From the results, it seems that documents from the same topic were closer to one
another than documents from different clusters. This conclusion remained
consistent upon increasing the sample, which supports the claim that documents
of the same topic will be more similar than to those from other topics.

