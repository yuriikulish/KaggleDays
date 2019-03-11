#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 09:38:01 2019

@author: kostya
"""

import pickle
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from model_params import params, params2
from tsvd_feats import create_tsvd_feats
from oof_prediction import add_test_oof


def load_train_test(oof = False, tfidf_svd = True, target='target_log', knn_feats=False):
    features = pickle.load(open('features/features.pkl','rb'))

    if knn_feats == True:
        exclude_features = ['price_mean_diff', 'addtocart','enc_model', 'enc_macro', 'enc_sub', 'enc_sub_macro']
    else:
        exclude_features = ['price_mean_diff', 'addtocart','enc_model', 'enc_macro', 'enc_sub', 'enc_sub_macro'] + \
            [col for col in features if "sum_sales_neighb_per" in col]

    features = pickle.load(open('features/features.pkl','rb'))
    train = pd.read_csv('features/train.csv')
    test = pd.read_csv('features/test.csv')

    if oof:
        all_df = pd.read_csv("oof_all.csv")
        train = pd.merge(train,all_df, on = ["sku_hash","month"])
        features = [q for q in features + ["oof"]]
        features = [q for q in features if q!="month"]
        test = add_test_oof(test)

    train[target] = np.log1p(train['target'])
    
    folds = pd.read_csv('features/folds.csv')
    
    outer_folds = []
    for i in range(5):
        train_idx = folds[folds['fold']!=i].index.tolist()
        test_idx = folds[folds['fold']==i].index.tolist()
        outer_folds.append((train_idx, test_idx))
    
    features = [c for c in features if c not in exclude_features]
    
    # TFIDF 
    if tfidf_svd:
        create_tsvd_feats()
        with open("features/tsvd_feats.pickle","rb") as f:
           tsvd_train_features, tsvd_test_features = pickle.load(f)
        
        train_svd_df = pd.DataFrame(tsvd_train_features)
        train_svd_df.columns = ['svd_{}'.format(c) for c in train_svd_df.columns]
    
        test_svd_df = pd.DataFrame(tsvd_test_features)
        test_svd_df.columns = ['svd_{}'.format(c) for c in test_svd_df.columns]
        
        train_svd = pd.concat((train, train_svd_df), axis=1)
        test_svd = pd.concat((test, test_svd_df), axis=1)
    
        features.extend(train_svd_df.columns)
        
        ds_train = lgb.Dataset(train_svd[features], train_svd[target].values)
    else:
        ds_train = lgb.Dataset(train[features], train[target].values)
    return train, ds_train,test, test_svd, outer_folds, train_svd, features

################################# Model 1 ##################################

train, ds_train,test, test_svd, outer_folds, train_svd, features = load_train_test(oof = False, tfidf_svd = True)

lgb_rounds = lgb.cv(params2, ds_train, num_boost_round = 3000, folds=outer_folds,
                    stratified = False, early_stopping_rounds = 100, verbose_eval=True)
print(lgb_rounds['rmse-mean'][-1], len(lgb_rounds['rmse-mean']))
lgb_model = lgb.train(params2, ds_train, num_boost_round=len(lgb_rounds['rmse-mean']))
train["prds"]  = lgb_model.predict(train_svd[features])
print(mean_squared_error(train["prds"],np.log1p(train['target'])))

y_predict = lgb_model.predict(test_svd[features])
test['target'] = np.expm1(y_predict)

test[['ID', 'target']].to_csv('submissions/subm_m1.csv', index=False, header=True)

##################################### Model 2 #####################################

train, ds_train,test, test_svd, outer_folds, train_svd, features = load_train_test(oof=True, tfidf_svd=True)

lgb_rounds = lgb.cv(params, ds_train, num_boost_round = 3000, folds=outer_folds,
                    stratified = False, early_stopping_rounds = 100, verbose_eval=True)
print(lgb_rounds['rmse-mean'][-1], len(lgb_rounds['rmse-mean']))
lgb_model = lgb.train(params2, ds_train, num_boost_round=len(lgb_rounds['rmse-mean']))
train["prds"]  = lgb_model.predict(train_svd[features])
print(mean_squared_error(train["prds"],np.log1p(train['target'])))

y_predict  = lgb_model.predict(test_svd[features])
test['target'] = np.expm1(y_predict)

test[['ID','target']].to_csv('submissions/subm_m2.csv', index=False, header=True)

##################################### Model 3 #####################################

train, ds_train,test, test_svd, outer_folds, train_svd, features = load_train_test(oof = True, tfidf_svd=True, knn_feats=True)

lgb_rounds = lgb.cv(params, ds_train, num_boost_round=3000, folds=outer_folds,
                    stratified=False, early_stopping_rounds=100, verbose_eval=True)
print(lgb_rounds['rmse-mean'][-1], len(lgb_rounds['rmse-mean']))
lgb_model = lgb.train(params2, ds_train, num_boost_round=len(lgb_rounds['rmse-mean']))
train["prds"] = lgb_model.predict(train_svd[features])
print(mean_squared_error(train["prds"],np.log1p(train['target'])))

y_predict = lgb_model.predict(test_svd[features])
test['target'] = np.expm1(y_predict)

test[['ID', 'target']].to_csv('submissions/subm_m3.csv', index=False, header=True)
