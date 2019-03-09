#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 17:43:20 2019

@author: kostya
"""
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from oof_prediction import oof_prediction
from knn_feats import create_knn_feats

# Frequency encoder for categorical

num_features = ['fr_FR_price']
sales_features = []
nav_features = []
cat_features = []
target = 'target'

cat_columns_target = []
cat_columns_count = ['function', 'sub_function', 'model', 'aesthetic_sub_line',
                      'macro_material', 'color', 'product_type', 'product_gender', 'macro_function']

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_sales = pd.read_csv('data/sales.csv')
df_navigation = pd.read_csv('data/navigation.csv')

df_train_test = df_train.append(df_test)

# Sum per zone
sales_per_zone = df_sales[['sku_hash', 'zone_number', 'sales_quantity']].\
                groupby(['sku_hash', 'zone_number']).sum().reset_index()
sales_pivot = sales_per_zone.pivot(index='sku_hash',columns='zone_number',values='sales_quantity')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['sum_sales_per_zone_{}'.format(i+1) for i in range(5)]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')

# Sum per country
sales_per_country = df_sales[['sku_hash', 'country_number', 'sales_quantity']].\
                groupby(['sku_hash', 'country_number']).sum().reset_index()
sales_pivot = sales_per_country.pivot(index='sku_hash',columns='country_number',values='sales_quantity')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['sum_sales_per_country_{}'.format(i) for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')


# Sum per day
sales_per_day = df_sales[['sku_hash', 'Date', 'sales_quantity']].\
                groupby(['sku_hash', 'Date']).sum().reset_index()
sales_pivot = sales_per_day.pivot(index='sku_hash',columns='Date',values='sales_quantity')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['sum_sales_per_{}'.format(i).lower() for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')

# Sum per day neighboor
sales_per_day_neigh = df_sales[['sku_hash', 'Date', 'sales_quantity']].\
                groupby(['sku_hash', 'Date']).sum().reset_index()
sales_pivot_neigh = sales_per_day.pivot(index='sku_hash',columns='Date',values='sales_quantity')
sales_pivot_neigh = sales_pivot.fillna(0)
sales_pivot_neigh.columns = ["sku_hash"] + ['sum_sales_neighb_per_{}'.format(i).lower() \
                             for i in sales_pivot_neigh.columns if i!="sku_hash"]

create_knn_feats()
knn = pd.read_csv("knn_feats.csv")

sales_pivot_neigh = pd.merge(sales_pivot_neigh, knn, on = "sku_hash")
sales_pivot_neigh["pt"] = "tst"

llist = []
for sku, knn_list in zip(sales_pivot_neigh["sku_hash"].values, sales_pivot_neigh["knn"].values):
    array_val = sales_pivot_neigh[sales_pivot_neigh["sku_hash"].isin(knn_list)].groupby("pt").sum()
    llist.append([sku] + list(array_val.values[0]/5))
sales_pivot_neigh2 = pd.DataFrame(llist, columns = sales_pivot_neigh.columns[:8])
sales_features.extend(list(sales_pivot_neigh.columns[1:8]))
df_train_test = df_train_test.merge(sales_pivot_neigh2, how='left', on='sku_hash')

#Sum per type
sales_per_type = df_sales[['sku_hash', 'type', 'sales_quantity']].\
                groupby(['sku_hash', 'type']).sum().reset_index()
sales_pivot = sales_per_type.pivot(index='sku_hash',columns='type',values='sales_quantity')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['sum_sales_per_{}'.format(i).lower() for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')


# launch day, launch month
launch_day = df_sales[df_sales['Date']=='Day_1']
sku_launch_day = launch_day.drop_duplicates(['sku_hash'])
le = LabelEncoder()
sku_launch_day['dow'] = le.fit_transform(sku_launch_day['day_transaction_date'])
sku_launch_day = sku_launch_day[['sku_hash', 'Month_transaction', 'dow']]
sku_launch_day.columns = ['sku_hash', 'launch_month', 'launch_dow']
df_train_test = df_train_test.merge(sku_launch_day, how='left', on='sku_hash')
sales_features.extend(['launch_month', 'launch_dow'])

# Total buzz per day
totalbuzz_per_day = df_sales[['sku_hash', 'Date', 'Impressions_6_day_before']].\
                groupby(['sku_hash', 'Date']).sum().reset_index()
sales_pivot = totalbuzz_per_day.pivot(index='sku_hash',columns='Date',values='Impressions_6_day_before')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['net_sant_per_{}'.format(i).lower() for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')

# Total impression per day
Impressions_per_day = df_sales[['sku_hash', 'Date', 'Impressions']].\
                groupby(['sku_hash', 'Date']).sum().reset_index()
sales_pivot = Impressions_per_day.pivot(index='sku_hash',columns='Date',values='Impressions')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['net_imp_per_{}'.format(i).lower() for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')

# Total impression per day
NetSentiment_per_day = df_sales[['sku_hash', 'Date', 'NetSentiment']].\
                groupby(['sku_hash', 'Date']).sum().reset_index()
sales_pivot = NetSentiment_per_day.pivot(index='sku_hash',columns='Date',values='NetSentiment')
sales_pivot = sales_pivot.fillna(0)
sales_pivot.columns = ['net_netsent_per_{}'.format(i).lower() for i in sales_pivot.columns]
sales_features.extend(list(sales_pivot.columns))
sales_pivot = sales_pivot.reset_index()
df_train_test = df_train_test.merge(sales_pivot, how='left', on='sku_hash')

# Add to cart features
add_to_cart = df_navigation[['sku_hash', 'addtocart']].groupby(['sku_hash']).sum().reset_index()
df_train_test = df_train_test.merge(add_to_cart, how='left', on='sku_hash')
nav_features.append('addtocart')
#AVG Price per product_type and gender
mean_price = df_train_test[['product_type', 'product_gender', 'macro_function', 'fr_FR_price']].\
        groupby(['product_type', 'product_gender', 'macro_function']).mean().reset_index()
mp_dict = {}
for mp in mean_price.iterrows():
    mp_dict[(mp[1]['product_type'],mp[1]['product_gender'],mp[1]['macro_function'])] = \
        mp[1]['fr_FR_price']
df_train_test['price_mean_diff'] = df_train_test.apply(lambda x: x['fr_FR_price']- \
                 mp_dict[(x['product_type'], x['product_gender'], x['macro_function'])],axis=1)

num_features.append('price_mean_diff')
#FILL NA
for c in cat_columns_count:
    df_train_test[c] = df_train_test[c].fillna('Unknown')
    
for c in cat_columns_count:
    cnt = df_train_test[['ID', c]].groupby(c).count()
    cnt_dict = {}
    for t in cnt.iterrows():
        cnt_dict[t[0]] = t[1]['ID']
    
    df_train_test['ce_{}'.format(c)]= df_train_test[c].apply(lambda x: cnt_dict[x])
    cat_features.append('ce_{}'.format(c))

features = num_features + cat_features + sales_features + nav_features + ["month"]

new_train = df_train_test[~df_train_test[target].isnull()][features+[target]+["sku_hash"]]
new_test = df_train_test[df_train_test[target].isnull()][features+['ID']+["sku_hash"]]

#OOF
new_train.to_csv('features/train.csv', index=False, header=True)
new_test.to_csv('features/test.csv', index=False, header=True)

oof_prediction(features)


pickle.dump(features, open('features/features.pkl','wb'))

###############################

####### knn_feats




