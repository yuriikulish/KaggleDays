# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:48:31 2019

@author: User
"""

import pandas as pd
from create_folds import create_folds
from model_params import params, params2
import pickle
from train_model import oof_train

def match_local(row):
    if row["month"] == 1:
        return row["oof1"]
    elif row["month"] == 2:
        return row["oof2"]
    else:
        return row["oof3"]

def oof_prediction(features, file = "oof_all.csv"):
    #features = pickle.load(open('features/features.pkl','rb'))
    
    for month in [1,2,3]:
        create_folds(1)
        create_folds(2)
        create_folds(3)
        create_folds(0,use_month_model = False)
    
    train1 = oof_train(params2, features, 1, target = 'target_log')
    train2 = oof_train(params2, features, 2, target = 'target_log')
    train3 = oof_train(params2, features, 3, target = 'target_log')
    
    pd.concat([train1[["sku_hash","oof","month"]],
               train2[["sku_hash","oof","month"]],
               train3[["sku_hash","oof","month"]]],0).to_csv(file,index = False)

def add_test_oof(test_svd):
    with open("test_preds1","rb") as f:
        pred_test = pickle.load(f)
    
    with open("test_preds2","rb") as f:
        pred_test2 = pickle.load(f)
    
    with open("test_preds3","rb") as f:
        pred_test3 = pickle.load(f)

    test_svd["oof1"] = pred_test
    test_svd["oof2"] = pred_test2
    test_svd["oof3"] = pred_test3
    test_svd["oof"] = test_svd.apply(lambda x : match_local(x),1)
    return test_svd
