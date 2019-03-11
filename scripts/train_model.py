# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:42:54 2019

@author: User
"""

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import numpy as np


def train_model(params, features, folds, ds_train):
    lgb_rounds = lgb.cv(params, ds_train, num_boost_round = 3000, folds=folds,
                    stratified = False,
            early_stopping_rounds = 100, verbose_eval=True)
    
    print(lgb_rounds['rmse-mean'][-1], len(lgb_rounds['rmse-mean']))

    lgb_model = lgb.train(params, ds_train, num_boost_round=len(lgb_rounds))
    return lgb_model


def oof_train(params, features, month, target='target_log'):
    train = pd.read_csv('features/train.csv')
    test = pd.read_csv('features/test.csv')
    features = [y for y in features if y!="month"]

    folds_first = pd.read_csv('features/folds_{}.csv'.format(month))
    folds_first.columns = ["index","fold"]
    folds_first.index = folds_first["index"]

    train1 = train[train["month"] == month]
    train1["oof"] = 0
    train1['target_log'] = np.log1p(train['target'])

    pred_test = []
    i = 0

    outer_folds_first = []
    for i in range(5):
        train_idx = folds_first[folds_first['fold'] != i].index.tolist()
        test_idx = folds_first[folds_first['fold'] == i].index.tolist()
        outer_folds_first.append((train_idx, test_idx))
    
    for train_idx, test_idx in outer_folds_first:
        print("Fold_{}".format(i))
        ds_fold_train = lgb.Dataset(train1.loc[train_idx][features], train1.loc[train_idx][target].values)
    
        lgb_model = lgb.train(params, ds_fold_train, num_boost_round=500)
    
        y_predict = lgb_model.predict(train1.loc[test_idx][features])
        y_predict_train = lgb_model.predict(train1.loc[train_idx][features])
    
        print("Valid error {}".format(mean_squared_error(list(train1.loc[test_idx][target].values), y_predict)))
        print("Train error {}".format(mean_squared_error(list(train1.loc[train_idx][target].values), y_predict_train)))
        
        train1.loc[test_idx, "oof"] = y_predict
        pred_test.append(lgb_model.predict(test[features]))
        i += 1
        train1["month"] = month
        
    with open("test_preds{}".format(str(month)), "wb") as f:
        pickle.dump(sum(pred_test)/5, f)

    return train1
