# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:41:35 2019

@author: User
"""

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd


def create_tsvd_feats():
    # ToDo: add correct field
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    tfv = TfidfVectorizer(min_df=1,  max_features=None, 
                          strip_accents='unicode', analyzer='word', token_pattern=r'\w+',
                          ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1)
    
    tfv.fit(pd.concat([train_df["en_US_description"], test_df["en_US_description"]], 0))
    vectors_train = tfv.transform(train_df["en_US_description"])
    vectors_test = tfv.transform(test_df["en_US_description"])
    print("TF IDF completed")
    
    tsvd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
    tsvd_vecs_train = tsvd.fit_transform(vectors_train)
    tsvd_vecs_test = tsvd.transform(vectors_test)
    
    with open("features/tsvd_feats.pickle", "wb") as f:
        pickle.dump((tsvd_vecs_train, tsvd_vecs_test), f)
