# Kaggle Days. Paris. 4th place solution

We used Python3 and a couple of useful libraries ;) to install them into your environment just execute:
```
pip3 install -r requirements.txt
```

## Features

Then we need to generate some features. Specifically, for each sku we generated: 
 - Total sales per zone
 - Total sales per country
 - Total sales per day
 - Total sales of 5 similar to current products in terms of product look (knn)
 - Total sales by type 
 - Product launch day of the week and month
 - Total impression and sentiment per day (not sure if we used this feature in final model)
 - Total addtocart
 - Average price per product type and gender
 - fr_FR_price
 - TFIDF on product description
 - Categorical features was encoded using count encoder

Besides that we used out of fold predictions for each month as a features for a model.

To generate features run:

 
```
python3 scripts/preprocessing.py
```

## Models training

We built three LightGBM models.

1st model. All features without out of folds and knn
2nd model. All features without knn
3rd model. All features.

To train models, and generate predictions run:

```
python3 scripts/lgbm_model.py
```

The predictions will be stored in submissions folder.

## Model ensemble 

We used a simple weighted average of the model predictions to generate a finale results.

```
python3 scripts/ensemble.py
```

The final submission file will be stored in submissions folder.

Feel free to contact us if you will have any questions regarding the solution - Yurii Kulish <yurii.kulish@gmail.com> or Konstantin Proskudin <proskudin@gmail.com>
