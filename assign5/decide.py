#!/usr/bin/python3

# Nicholas Antonov and Pat Grasso worked together on this assignment

import nltk, csv, math, re, pandas
import numpy as np
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from tficf import TficfVectorizer

from printtree import tree_to_code

warnings.simplefilter("ignore", np.VisibleDeprecationWarning)

def report(results, n_top=3):
    """
    Report the top n hyperparameter configurations for this set
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results["mean_test_score"][candidate],
                results["std_test_score"][candidate]
            ))
            print("Parameters: {0}".format(results["params"][candidate]))
            print()


samples = pandas.read_csv("serious.csv", encoding="latin1")
samples = samples[samples["SYMPTOM_TEXT"].notnull()]

# Partition the data into training/testing sets
train_samples = samples.sample(frac=0.5)
test_samples = samples.drop(train_samples.index)#.sample(frac=0.05)
#test_yes = test_samples[test_samples["SERIOUS"] == "Y"]
#test_no = test_samples[test_samples["SERIOUS"] == "N"].sample(test_yes.shape[0])
#test_samples = pandas.concat((test_no, test_yes))

print("Training on {} samples".format(train_samples.shape[0]))

vectorizers = {
    "tfidf": TfidfVectorizer(max_df=1, max_features=1000,
                             min_df=1, stop_words='english',
                             use_idf=True, ngram_range=(1,3)),
    "count": CountVectorizer(stop_words="english", ngram_range=(1,3),
                             max_features=1000),
    "tficf": TficfVectorizer()
}


# Pick vectorization method
vectorizer = vectorizers["tficf"]
word_matrix = vectorizer.fit_transform(
    train_samples["SYMPTOM_TEXT"],
    train_samples["SERIOUS"]).toarray()
test_matrix = vectorizer.transform(
    test_samples["SYMPTOM_TEXT"],
    test_samples["SERIOUS"]).toarray()

exit()


# Set up decision tree
decision_tree = DecisionTreeClassifier()
clf = GridSearchCV(
    decision_tree,
    { "max_depth": [5, 6, 7],
      "min_samples_leaf": [1, 4, 5, 6, 7] },
    n_jobs = 8
)

baseline = DummyClassifier(strategy="prior")

clf.fit(word_matrix, train_samples["SERIOUS"])
baseline.fit(word_matrix, train_samples["SERIOUS"])

# Report top hyperparam configs
report(clf.cv_results_)

# Print tree
#tree_to_code(clf.best_estimator_, vectorizer.get_feature_names())
export_graphviz(clf.best_estimator_,
                out_file = "tree.dot",
                feature_names = vectorizer.get_feature_names(),
                class_names = ["N", "Y"],
                filled = True,
                impurity = False,
                proportion = True)

# Get feature importances
feature_importances = sorted(zip(
    clf.best_estimator_.feature_importances_,
    vectorizer.get_feature_names()
), reverse=True)

# Print the top N important features
for importance, feature in feature_importances[:10]:
    feature = (feature[:27] + "...") if len(feature) > 30 else feature
    print("{:30s}: {}".format(feature, importance))

# Print classifier's accuracy on the test set
print("score   :", clf.score(test_matrix, test_samples["SERIOUS"]))
print("baseline:", baseline.score(test_matrix, test_samples["SERIOUS"]))



