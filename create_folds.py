#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:17:25 2019

@author: kostya
"""

import pandas as pd
from sklearn.model_selection import KFold

def create_folds(month, use_month_model = True):
    df_train = pd.read_csv('data/train.csv')
    df_train['fold'] = None

    if use_month_model == True:
        df_train = df_train[df_train["month"] == month]
       
        random_state = 345
        kf = KFold(5, random_state=random_state, shuffle=True)
        
        
        skus = df_train['sku_hash'].unique()
        for i, (train_index, test_index) in enumerate(kf.split(skus)):
            #train_sku = skus[train_index]
            test_sku = skus[test_index]
            df_train['fold'][df_train['sku_hash'].apply(lambda x: x in test_sku)] = i
            
            
        df_train['fold'].to_csv('features/folds_{}.csv'.format(str(month)), header=True)
    else: 
        random_state = 7777
        kf = KFold(5, random_state=random_state, shuffle=True)
        
        
        skus = df_train['sku_hash'].unique()
        for i, (train_index, test_index) in enumerate(kf.split(skus)):
            #train_sku = skus[train_index]
            test_sku = skus[test_index]
            df_train['fold'][df_train['sku_hash'].apply(lambda x: x in test_sku)] = i
        
        
    df_train['fold'].to_csv('features/folds.csv', header=True)
