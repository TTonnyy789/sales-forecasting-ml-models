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
from keras.models import Sequential, LSTM



## Build a class called SalesForecaster for the purpose of making predictions faster
class SalesForecaster:
    ## Load the data from your local system
    df_train_complete = pd.read_csv("/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/complete_training_data.csv")


    ## Initialising the class with the store number and training data and validation time period
    def __init__(self, store_number, start_date='2013-01-01', end_date='2015-07-31'):
        self.store_number = store_number
        self.start_date = start_date
        self.end_date = end_date
        self.df_train_complete = pd.read_csv("/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/complete_training_data.csv")
    
    def filter_training_data_monthly(self):
        filtered_df = self.df_train_complete[self.df_train_complete['Store'] == self.store_number]
        filtered_df = filtered_df.resample('M').mean()
        filtered_df['Sales'] = filtered_df['Sales'].astype(int)
        return filtered_df


# df_check_period13 = complete_df.loc[(complete_df['Store'] == store_number) & (complete_df.index >= start_missing_date) & (complete_df.index <= end_missing_date)]


    ## Create a multi-output linear regression model
    def linear_regression_model(self):
        pre_x_train = self.filter_training_data_monthly().loc[self.filter_training_data_monthly().index < self.train_end_date]
        X_train = pre_x_train.drop(['Sales'], axis=1)
        y_train = self.filter_training_data_monthly()['Sales']
        X_test = self.filter_training_data_monthly().drop(['Sales'], axis=1)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

        



