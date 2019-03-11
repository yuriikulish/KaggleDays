# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:26:34 2019

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

df = pd.read_csv("data/vimages.csv")
list_sku = df["sku_hash"].values


def find_top_similarities(arr):
    arr2 = list(reversed(np.argsort(arr)))
    return list_sku[arr2[1:6]]


def create_knn_feats():
    A_sparse = sparse.csr_matrix(df[[e for e in df.columns if e != "sku_hash"]])
    similarities = cosine_similarity(A_sparse)
    print('pairwise dense output:\n {}\n'.format(similarities))

    df["knn"] = [find_top_similarities(w) for w in similarities]
    df[["sku_hash", "knn"]].head()

    df["knn"] = [find_top_similarities(w) for w in similarities]
    df[["sku_hash", "knn"]].to_csv("knn_data.csv", index=False)
