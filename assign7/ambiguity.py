
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import word_tokenize

ambiguous_word = "hand"

sentences = [
    ("hand me the spoon, please"            , wn.synset("pass.v.05")),
    ("he put his hand on top of his head"   , wn.synset("hand.n.01"))
]

print("Ambiguous Word:  ", ambiguous_word)

for sent, sense in sentences:
    synset = lesk(word_tokenize(sent), ambiguous_word)
    print()
    print("Sentence:    ", sent)
    print("True Defn:   ", sense.name(), "-", sense.definition())
    print("Predict Defn:", synset.name(), "-", synset.definition())
    print("Similarity:  ", synset.wup_similarity(sense))
