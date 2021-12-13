import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans


# vocab created using tf-idf lib in Clojure
vocabulary = json.load(open("../resources/parliament/vocabulary.json"))

# using combined CSV file created using tech.ml.dataset in Clojure
df = pd.read_csv('../resources/parliament/data.csv')

# split into environment and immigration dataframes
#environment_df = df[(df['Subject-1'] == 'Environment') | (df['Subject-2'] == 'Environment')]
#immigration_df = df[(df['Subject-1'] == 'Immigration') | (df['Subject-2'] == 'Immigration')]

#corpus = df[["Lemma"]].replace('', np.nan).dropna().to_numpy().flatten()

# applying transformation
#tfidf_corpus = vectorizer.fit_transform(corpus)


def by_subject(df, subject):
    return df[(df['Subject-1'] == subject) | (df['Subject-2'] == subject)]


def text_data(df, column):
    return df[[column]].replace('', np.nan).dropna().to_numpy().flatten()


def mk_cluster(X):
    # Seems like AffinityPropagatin is a no-go: https://stackoverflow.com/questions/35901718/sklearn-affinitypropagation-memoryerror
    return AffinityPropagation().fit(X=X)
    #return MiniBatchKMeans(n_clusters=12).fit(X=X)


def run_clustering(df, column, subject):
    #vectorizer = HashingVectorizer()
    #vectorizer = CountVectorizer(vocabulary=vocabulary)
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    return mk_cluster(vectorizer.fit_transform(text_data(by_subject(df, subject), column)))

"""
c = run_clustering(df, 'Lemma', 'Environment')
"""




