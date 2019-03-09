# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:09:19 2019

@author: User
"""

params = {}
params['learning_rate'] = 0.05 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'huber'
params['metric'] = 'l2_root'      
params['verbose'] = 0
params['feature_fraction_seed'] = 2
params['bagging_seed'] = 3
params["num_threads"] = 6

params2 = {}
params2['learning_rate'] = 0.05 # shrinkage_rate
params2['boosting_type'] = 'gbdt'
params2['objective'] = 'huber'
params2['metric'] = 'l2_root'      
params2['sub_feature'] = 0.7
params2['min_data'] = 200
params2['lambda_l2'] = 5.
params2['verbose'] = 0
params2['feature_fraction_seed'] = 2
params2['bagging_seed'] = 3
params2["num_threads"] = 6
