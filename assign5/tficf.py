
import numpy as np
import pandas, scipy, itertools

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TficfVectorizer(CountVectorizer):
    """
    Performs tf-idf on a set of documents, except the inverse document
    frequency is computed using the entire set of documents for each
    class as a single "document"
    """

    def __init__(self, norm="l2", use_idf=False, smooth_idf=True,
                 sublinear_tf=False, **kwargs):
        super(TficfVectorizer, self).__init__(**kwargs)
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y):
        indices = X.index
        X = super(TficfVectorizer, self).fit_transform(X)
        self._tfidf.fit(X)
        self._icf = self._icf(X.toarray(), y)
        tfs = pandas.DataFrame(self._tfidf.transform(X).todense(),
                               index = indices)

        groups = []
        for name, group in pandas.groupby(tfs, y):
            groups += [group * self._icf.ix[name]]

        return pandas.concat(groups)

    def _icf(self, X, y):
        X = pandas.DataFrame(X)
        y = pandas.Series(y)

        grouped = pandas.groupby(X, y)
        class_count = grouped.sum().divide(grouped.size(), axis="index")
        class_importance = (class_count / class_count.sum()).fillna(0)
        return class_importance

    def transform(self, X, y):
        indices = X.index
        X = super(TficfVectorizer, self).transform(X)
        tfs = pandas.DataFrame(self._tfidf.transform(X).todense(),
                               index = indices)

        groups = []
        for name, group in pandas.groupby(tfs, y):
            groups += [group * self._icf.ix[name]]

        return pandas.concat(groups)


