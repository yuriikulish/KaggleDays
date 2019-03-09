# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:46:20 2019

@author: User
"""

import lightgbm as lgb
import pandas as pd
import math
import numpy as np

def is_weekday(x):
    if x in ["Sunday","Saturday"]:
        return 0
    else:
        return 1

def apply_row_prediction(row):
    if row["month"] in [1,3]:
        return row["sales_quantity"] * 4 + row["sales_quantity"]/7 * 3 
    else :
        return row["sales_quantity"] * 4

def apply_prediction(df):
    df = df.groupby("sku_hash")["sales_quantity"].sum().reset_index()
    df["key"] = "join_key"
    df = pd.merge(df, pd.DataFrame({"key" : ["join_key"] * 3,"month": [1,2,3]}), on = "key")
    df["sales_quantity"] = df.apply(apply_row_prediction,1)
    return df

def apply_country_prediction(df):
    df = df.groupby(["sku_hash","country_number"])["sales_quantity"].sum().reset_index()
    df["key"] = "join_key"
    df = pd.merge(df, pd.DataFrame({"key" : ["join_key"] * 3,"month": [1,2,3]}), on = "key")
    df["sales_quantity"] = df.apply(apply_row_prediction,1)
    
    df = df.groupby(["sku_hash","month"])["sales_quantity"].sum().reset_index()    
    return df

def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def calculate_error(df, pred_df):
    j_df = pd.merge(df,pred_df, on = ["sku_hash","month"])
    print("RMLSE error {}".format(rmsle(j_df["target"], j_df["sales_quantity"])))

#nav_df[nav_df["sku_hash"] == "ed4c7471eac7e8c6e6718364c2b6e75462eeb47c"]
#test_df[test_df["sku_hash"] == "ed4c7471eac7e8c6e6718364c2b6e75462eeb47c"]
#train_df[train_df["sku_hash"] == "ed4c7471eac7e8c6e6718364c2b6e75462eeb47c"]
#sales_df[sales_df["sku_hash"]=="ed4c7471eac7e8c6e6718364c2b6e75462eeb47c"]

nav_df = pd.read_csv("./data/navigation.csv")
sales_df = pd.read_csv("./data/sales.csv")
test_df = pd.read_csv("./data/test.csv")
train_df = pd.read_csv("./data/train.csv")
nav_df["is_weekend"] = nav_df["day_visit"].apply(lambda x : is_weekday(x),1)

###### apply day prediction #############
    
sales_date_df = sales_df.groupby(["sku_hash","Date"])["sales_quantity"].sum().reset_index()
sales_date_country_df = sales_df.groupby(["sku_hash","Date","country_number"])["sales_quantity"].sum().reset_index()

sales_full_df = sales_df.groupby(["sku_hash"])["sales_quantity"].sum().reset_index()
pred_df = apply_prediction(sales_full_df)

pred_country_df = apply_country_prediction(sales_date_country_df)    
calculate_error(train_df, pred_country_df)

################################ Salse approach ###################################################

month_used = 1
### first_month_pred
train_month_df = train_df[train_df["month"] == month_used]
test_month_df = test_df[test_df["month"] == month_used]

X_train = pd.merge(sales_df,train_month_df, on = "sku_hash")
X_test = pd.merge(sales_df,test_month_df, on = "sku_hash")

####

nav_df = pd.read_csv("./data/navigation.csv")
nav_per_day = pd.pivot_table(nav_df, values='page_views', index=["sku_hash"], columns=['Date'], aggfunc=np.sum, fill_value=0).reset_index()
nav_per_day.columns = ["sku_hash"] + ["nav_day_" + str(i) for i in range(1,8)]

nav_per_weekday = pd.pivot_table(nav_df, values='page_views', index=["sku_hash"], columns=['is_weekend'], aggfunc=np.sum, fill_value=0).reset_index()
nav_per_weekday.columns = ["sku_hash"] + ["nav_weekday_" + str(i) for i in range(1,3)]

nav_per_zone_number = pd.pivot_table(nav_df, values='page_views', index=["sku_hash"], columns=['website_version_zone_number'], aggfunc=np.sum, fill_value=0).reset_index()
nav_per_zone_number.columns = ["sku_hash"] + ["nav_zone_" + str(t) for t in nav_per_zone_number.columns[1:]]

nav_per_country_number = pd.pivot_table(nav_df, values='page_views', index=["sku_hash"], columns=['website_version_country_number'], aggfunc=np.sum, fill_value=0).reset_index()
nav_per_country_number.columns = ["sku_hash"] + ["nav_country_" + str(t) for t in nav_per_country_number.columns[1:]]

f_df = pd.merge(nav_per_day, nav_per_weekday, on = "sku_hash")
f_df = pd.merge(f_df, nav_per_zone_number, on = "sku_hash")
f_df = pd.merge(f_df, nav_per_country_number, on = "sku_hash")

f_df.to_csv("nav_features_sku.csv", index = False)

#################################################################

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
tfv_text = TfidfVectorizer(min_df=5,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w+',
            ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

full_text = pd.concat([train_df["en_US_description"], test_df["en_US_description"]],0).drop_duplicates()

tfv_text.fit(full_text)
vectors = tfv_text.transform(full_text)

tsvd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
tsvd_vecs = tsvd.fit_transform(vectors)

tsvd_train_features = tsvd.transform(tfv_text.transform(train_df["en_US_description"]))
tsvd_test_features = tsvd.transform(tfv_text.transform(test_df["en_US_description"]))

import pickle
with open("tsvd_feats.pickle","wb") as f:
    pickle.dump((tsvd_train_features, tsvd_test_features),f)

with open("tsvd_feats.pickle","rb") as f:
    tsvd_train_features, tsvd_test_features = pickle.load(f)

full_df = pd.concat([train_df, test_df], 0)

full_df[["macro_function","sub_function","aesthetic_sub_line","product_type","sku_hash","model"]].to_csv("qqq.csv", index = False)

def convert_to_dict(df, col1, col2):
    ddict = {}
    for itr, itr2 in df[[col1,col2]].values:
        ddict[itr] = itr2
    return ddict

def category_encoding(train_df, sales_df):
    sales_df2 = sales_df.groupby(["sku_hash"])["sales_quantity"].sum().reset_index()
    sales_df3 = pd.merge(sales_df2, 
                         train_df[["sku_hash","product_type","macro_function","sub_function","model"]].drop_duplicates(),
                         on = "sku_hash")

    sales_df3_model = sales_df3.groupby(["product_type","model"])["sales_quantity"].sum()
    sales_df3_macro = sales_df3.groupby(["product_type","macro_function"])["sales_quantity"].sum()
    sales_df3_sub = sales_df3.groupby(["product_type","sub_function"])["sales_quantity"].sum()
    sales_df3_macro_sub = sales_df3.groupby(["macro_function","sub_function"])["sales_quantity"].sum()

    sales_df3_model2 = sales_df3_model.groupby(level=0).apply(lambda x:
                                                     100 * x / float(x.sum())).reset_index()
    sales_df3_macro2 = sales_df3_macro.groupby(level=0).apply(lambda x:
                                                     100 * x / float(x.sum())).reset_index()
    sales_df3_sub2 = sales_df3_sub.groupby(level=0).apply(lambda x:
                                                     100 * x / float(x.sum())).reset_index()
    sales_df3_macro_sub2 = sales_df3_macro_sub.groupby(level=0).apply(lambda x:
                                                     100 * x / float(x.sum())).reset_index()
        
    macro_d = convert_to_dict(sales_df3_macro2,"macro_function", "sales_quantity") 
    model_d = convert_to_dict(sales_df3_model2,"model", "sales_quantity") 
    sub_d = convert_to_dict(sales_df3_sub2,"sub_function", "sales_quantity") 
    macro_sub_d = convert_to_dict(sales_df3_macro_sub2,"sub_function", "sales_quantity") 
    return macro_d, model_d, sub_d, macro_sub_d

macro_d, model_d, sub_d, macro_sub_d = category_encoding(train_df, sales_df)

train_df["enc_model"] = train_df["model"].apply(lambda x : model_d[x] if x in model_d.keys() else 0,1)
train_df["enc_macro"] = train_df["macro_function"].apply(lambda x : macro_d[x] if x in macro_d.keys() else 0,1)
train_df["enc_sub"] = train_df["sub_function"].apply(lambda x : sub_d[x] if x in sub_d.keys() else 0,1)
train_df["enc_sub_macro"] = train_df["sub_function"].apply(lambda x : macro_sub_d[x] if x in macro_sub_d.keys() else 0,1)

test_df["enc_model"] = test_df["model"].apply(lambda x : model_d[x] if x in model_d.keys() else 0,1)
test_df["enc_macro"] = test_df["macro_function"].apply(lambda x : macro_d[x] if x in macro_d.keys() else 0,1)
test_df["enc_sub"] = test_df["sub_function"].apply(lambda x : sub_d[x] if x in sub_d.keys() else 0,1)
test_df["enc_sub_macro"] = test_df["sub_function"].apply(lambda x : macro_sub_d[x] if x in macro_sub_d.keys() else 0,1)

features = ["enc_model","enc_macro","enc_sub","enc_sub_macro"]

train_df[features].to_csv("train_enc.csv", index = False)
test_df[features].to_csv("test_enc.csv", index = False)


df = pd.read_csv("data/vimages.csv")
list_sku = df["sku_hash"].values

from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

A_sparse = sparse.csr_matrix(df[[e for e in df.columns if e!="sku_hash"]])

similarities = cosine_similarity(A_sparse)
print('pairwise dense output:\n {}\n'.format(similarities))

def find_top_similarities(arr):
    arr2 = list(reversed(np.argsort(arr)))
    return list_sku[arr2[1:6]]

df["knn"] = [find_top_similarities(w) for w in similarities]

with open("knn.pickle","wb") as f:    
    pickle.dump(df[["sku_hash","knn"]],f)
    #df[["sku_hash","knn"]].to_csv("knn_data.csv", index = False)    
    
    
    