#%%#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


## Build a class called SalesForecaster for the purpose of making predictions faster
class SalesForecaster:
    ## Load the data from your local system
    product = pd.read_csv("/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/processed_training_data.csv")

    ### ...
    product['product_type'] = product['product_type'].astype('category')
    ## Create a dictionary for storing the store-product data sets
    segmented_data = {}

    #3 Function to process each segment if the data starts with zero sales units
    def process_segment(group):
        # Remove initial zero sales
        first_non_zero_index = group['sales'].ne(0).idxmax()
        return group.loc[first_non_zero_index:]
    # Grouping the data by store and product
    for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
        processed_group = process_segment(group)
        segmented_data[(store, product_type)] = processed_group[['sales', 'special_offer', 'id', 'store_nbr']]
    
    def __init__(self, store_number, train_end_date='2014-07-31', validation_end_date='2015-07-31'):
        self.store_number = store_number
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        



