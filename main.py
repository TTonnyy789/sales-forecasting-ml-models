#%%#
### Step 1  ######################################################################
### Import essential library and load the data


## Import the library
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw, dtw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_predict
from nixtlats import NixtlaClient
from nixtlats.date_features import CountryHolidays
from pytorch_forecasting import TimeSeriesDataSet
from keras.models import Sequential, save_model, load_model, save_model
from keras.layers import Dense, LSTM, Embedding, Input
from statsmodels.tools.eval_measures import rmse, rmspe
from tensorflow.keras.optimizers import Adam


## Read the data 
df_store = pd.read_csv('/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/DA2024_stores.csv')

## Original training data set
# df_train = pd.read_csv('/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/DA2024_train.csv')

## Imputed training data set by implemented TImeGPT
df_train_complete = pd.read_csv('/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/complete_training_data.csv')


df_test = pd.read_csv('/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/DA2024_test.csv')


## Overview of the data
print("\n • Overview info of the store information")
print(" ")
print(df_store.info())
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train_complete.info())
print("------------------------------------------------")

print("\n • Training Data Set \n")
print(df_train_complete.head())
print("------------------------------------------------")
print("\n • Testing Data Set \n")
print(df_test.head())

## 01/01/2013 - 31/07/2015 -> Training Data, total 942 days ideally
## 01/08/2015 - 17/09/2015 -> Testing Data, total 48 days ideally

###-------------------------------------------------------------------------------



#%%#
### Step 2  ######################################################################
### Data Preprocessing, for basic missing value across all stores


## Deal with the missing values
store_missing = df_store.isnull().sum()
# train_missing = df_train.isnull().sum()


## Convert datetime into correct datetime format also configure the index into the datetime, only sutiable for the original training data
df_train_complete['Date'] = pd.to_datetime(df_train_complete['Date'], dayfirst=True)
df_train_complete.set_index('Date', inplace=True)
## Set a new column for duplicating the datetime index called Date_Column
df_train_complete['Date_Column'] = df_train_complete.index
# df_train_complete['Date_Column'] = df_train_complete.index

df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True)
df_test.set_index('Date', inplace=True)
df_test['Date_Column'] = df_test.index


## Reverse the index of training data and testing data
# df_train_processed = df_train.iloc[::-1]
df_test_processed = df_test.iloc[::-1]


## Check wether there are missing values in datetime index
# missing_dates_by_store = {}
# missing_data_stores_list = []

# ## Loop through each store number
# for i in range(1, 1116):  
#     store_data = df_train_processed[df_train_processed['Store'] == i]
    
#     ## Check if there is any data for the store; if not, continue to the next iteration
#     if store_data.empty:
#         missing_dates_by_store[i] = "No data available"
#         continue
#     ## Generate a complete date range for the period the store has data
#     expected_dates = pd.date_range(start=store_data.index.min(), end=store_data.index.max(), freq='D')    
#     ## Identify missing dates by finding the difference
#     actual_dates = store_data.index.unique()  # Ensure there are no duplicate dates
#     missing_dates = expected_dates.difference(actual_dates)
#     ## Store the missing dates in the dictionary
#     missing_dates_by_store[i] = missing_dates


# ## Print the missing date time for the stores
# for store, dates in missing_dates_by_store.items():
#     if not dates.empty:
#         missing_data_stores_list.append(store)
#         print(f"Store {store} has missing dates: {dates}")
#     else:
#         print(f"Store {store} has no missing dates.")
# print("------------------------------------------------ \n")
# print(f"Stores with missing data: {missing_data_stores_list}")


## Check the date range of the training data
start_date = df_train_complete.index.min()
end_date = df_train_complete.index.max()
print("The date range of the training data is:")
print("The dataset starts on:", start_date)
print("The dataset ends on:", end_date)
print('------------------------------------------------\n')


## Overview of the data again
print("\n • Overview info of the store information")
print(" ")
print(df_store.info())
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train_complete.info())
print("------------------------------------------------")

print("\n • Training Data Set \n")
print(df_train_complete)
print("------------------------------------------------")
print("\n • Testing Data Set \n")
print(df_test.head())

###-------------------------------------------------------------------------------



#%%#
### Step 2-1  ######################################################################
## Check the updated training data and missing values among the store information and training data
        

# ## Filter the missing data stores, and fill the missing value, firstly indentify the range of datetime for each store
# date_range = pd.date_range(df_train_processed.index.min(), df_train_processed.index.max())


# # Generate a DataFrame that contains all dates within the range for each store
# unique_stores = df_train['Store'].unique()
# date_store_combinations = pd.MultiIndex.from_product([date_range, unique_stores], names=['Date', 'Store'])
# complete_df = pd.DataFrame(index=date_store_combinations).reset_index()

# # Merge with the original data to fill in missing dates
# complete_df = complete_df.merge(df_train.reset_index(), on=['Date', 'Store'], how='left')
# complete_df.set_index('Date', inplace=True)
# print(complete_df)


## Check wether there are missing values in datetime index
missing_dates_by_store = {}
missing_data_stores_list2 = []

## Loop through each store number
for i in range(1, 1116):  
    store_data = df_train_complete[df_train_complete['Store'] == i]
    ## Check if there is any data for the store; if not, continue to the next iteration
    if store_data.empty:
        df_train_complete[i] = "No data available"
        continue
    # ## Generate a complete date range for the period the store has data
    # expected_dates = pd.date_range(start=store_data.index.min(), end=store_data.index.max(), freq='D')    
    # ## Identify missing dates by finding the difference
    # actual_dates = store_data.index.unique()  # Ensure there are no duplicate dates
    # missing_dates = expected_dates.difference(actual_dates)
    # ## Store the missing dates in the dictionary
    # df_train_complete[i] = missing_dates


## Print the missing date time for the stores
for store, dates in missing_dates_by_store.items():
    if not dates.empty:
        missing_data_stores_list2.append(store)
        print(f"Store {store} has missing dates: {dates}")
    else:
        print(f"Store {store} has no missing dates.")
print("------------------------------------------------ \n")
if missing_data_stores_list2 != []:
    print(f"Stores with missing data: {missing_data_stores_list2}")
else:
    print("There are no missing data in the training data.")


## Overview of the data
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train_complete.info())
print("------------------------------------------------ \n")
print("\n • Processed training Data Set \n")
print(df_train_complete.head())
print("------------------------------------------------")
print("\n • Processed testing Data Set \n")
print(df_test_processed.head())


###-------------------------------------------------------------------------------



#%%#
### Step 3  ######################################################################
### Data Imputation for those stores with missing values between '2014-07-01' and '2014-12-31' using Nixtla TimeGPT


## Print() the store number with missing values between '2014-07-01' and '2014-12-31'
# print("------------------------------------------------")
# print("\n • Store number with missing values between '2014-07-01' and '2014-12-31' \n")
# print(missing_data_stores_list)


# ## Define the date range for filled vales for forecasting
# start_date = '2013-01-01'
# end_date = '2014-06-30'

# ## Define the date range for missing values
# start_missing_date = '2014-07-01'
# end_missing_date = '2014-12-31'

# len_missing_date = len(pd.date_range(start_missing_date, end_missing_date, freq='D'))
# lenth = 184


## Approach 1: Using TimeGPT to impute the missing values
# key_test = 'nixt-eg8GKBAEdAFGkdp8OVfZBdH2jfwt3xQFRsQA2kXyeLAK8NfJt0LybXSgJ7olV9FAUe6nxLjDRdn49EmP'

# key1 = 'nixt-Vw0Kq21wmUVzwmnWKi4UagShcAxHI1Gz3jY4EnIxXoegNJqSIJFxEhfzQTRHtBVqx2oYqpFQ14qDZcPY'
# key2 = 'nixt-Vw0Kq21wmUVzwmnWKi4UagShcAxHI1Gz3jY4EnIxXoegNJqSIJFxEhfzQTRHtBVqx2oYqpFQ14qDZcPY'
# key3 = 'nixt-Of8rRNd7mo487ePPMuypjdFGDGRfbo2hwFMhsGYx4LBlopEZmnFi8i6rUXuvW2B4ezxHPp4Idmd76vEq'
# key4 = 'nixt-AoBrgLMvVSpfy5UwSigbDxuVqjOk4Lhu69vHS1SYOu9wpMRSz4vQYawfNYS4pLPaW5FX5870qqokt7HV'
# key5 = 'nixt-VErqALA4bwZVgAhMMnDoHavkGza8h53Nr9GbjRAgfAlg9cP9XvBNVeVLA55FLyc8Z0C2Jg57VPL9mM85'
# key6 = 'nixt-odlff4mIDpfEzVS9ohlBbAqgQqf6R8FT2H5yWa5sqF3gwz9iN7gQ6HBsWuZttIcW4MRDsERQXMKyg9zv'
# key7 = 'nixt-eqgt5P8DBOhCCt81EQ8TABXMDTRwAVoabEjswqdDktcJmWa7AmZ4wJoORiN3y9Ei9mhwMu5wfDDrco6c'
# key8 = 'nixt-eUZMR3D9b1neXCQ7IiiBEa7ipoj1XBLwySvCQhHchVhQtrW6kAuKYqE8iNH5WPbNdXHCGab4alEnFK2a'

# nixtla_client = NixtlaClient(api_key = key_test)

# nixtla_client.validate_api_key()

# missing_list_test = [13] 
# missing_list_1 = [20, 22, 32, 36, 41, 46, 51, 52, 58, 72, 76, 81, 89, 99, 100, 108, 113, 115, 127, 129, 132, 136, 137, 139, 144, 145, 149, 155, 159]
# missing_list_2 = [164, 165, 172, 174, 181, 183, 186, 190, 191, 192, 204, 215, 218, 231, 243, 258, 263, 275, 277, 279, 283, 284, 287, 288, 298, 306, 317]
# missing_list_3 = [342, 345, 348, 365, 385, 399, 407, 412, 413, 420, 427, 428, 429, 430, 434, 457, 471, 477, 485, 490, 492, 498, 500, 501, 512, 514]
# missing_list_4 = [518, 522, 534, 539, 540, 542, 547, 555, 571, 573, 575, 587, 598, 604, 611, 612, 619, 620, 629, 633, 636, 637, 638, 644, 646, 650]
# missing_list_5 = [660, 669, 670, 671, 677, 684, 694, 701, 702, 706, 710, 711, 712, 716, 719, 736, 739, 744, 750, 766, 771, 775, 778, 797, 804, 805, 806]
# missing_list_6 = [815, 820, 825, 842, 851, 858, 859, 879, 884, 890, 893, 900, 902, 903, 904, 909, 915, 919, 920, 932, 941, 952, 974, 977, 989, 1000]
# missing_list_7 = [1004, 1009, 1012, 1019, 1027, 1038, 1041, 1049, 1056, 1065, 1067, 1080, 1092, 1094, 1102, 1104, 1107, 1109]
# missing_list_8 = []



## Select target stores with missing values between '2014-07-01' and '2014-12-31'
# store_number = 13

# df_target_period13 = complete_df.loc[(complete_df['Store'] == store_number) & (complete_df.index >= start_date) & (complete_df.index <= end_date)]

# ## Ensure the data is complete for the forecasting
# print(df_target_period13)




#%%#
### Step 3-1  ######################################################################
## Use API key1 for the list1 stores to forecast the missing values, after Nixtla API checking, check the empty period for the store

# nixtla_client = NixtlaClient(api_key = key8)

# for i in missing_list_test:
#     df_target_period = complete_df.loc[(complete_df['Store'] == i) &
#                                        (complete_df.index >= start_date) &
#                                        (complete_df.index <= end_date)]
    
#     predicted_sales = nixtla_client.forecast(
#                 df=df_target_period, 
#                 h=184, 
#                 freq='D', 
#                 time_col='Date_Column', 
#                 target_col='Sales',
#                 finetune_steps=180,
#                 model='timegpt-1-long-horizon',
#                 finetune_loss='mae'
#                 )
    
#     predicted_sales['Date_Column'] = pd.to_datetime(predicted_sales['Date_Column'])
#     predicted_sales.set_index('Date_Column', inplace=True)
#     predicted_sales.index.name = 'Date'

#     for date, row in predicted_sales.iterrows():
#         if date in complete_df[(complete_df['Store'] == i)].index:
#             complete_df.loc[(complete_df['Store'] == i) & (complete_df.index == date), 'Sales'] = row['TimeGPT']



# ## Check the updated data
# store_number = 13

# df_check_period13 = complete_df.loc[(complete_df['Store'] == store_number) & (complete_df.index >= start_missing_date) & (complete_df.index <= end_missing_date)]

# ## Ensure the data is complete for the forecasting
# print(df_check_period13)
# print(complete_df)


###-------------------------------------------------------------------------------

#%%#
### Step 3-2  ######################################################################
## FInish the rest of other store list1


# nixtla_client = NixtlaClient(api_key = key1)
# for i in missing_list_1:
#     df_target_period = complete_df.loc[(complete_df['Store'] == i) &
#                                        (complete_df.index >= start_date) &
#                                        (complete_df.index <= end_date)]
    
#     predicted_sales = nixtla_client.forecast(
#                 df=df_target_period, 
#                 h=184, 
#                 freq='D', 
#                 time_col='Date_Column', 
#                 target_col='Sales',
#                 finetune_steps=180,
#                 model='timegpt-1-long-horizon',
#                 finetune_loss='mae'
#                 )
    
#     predicted_sales['Date_Column'] = pd.to_datetime(predicted_sales['Date_Column'])
#     predicted_sales.set_index('Date_Column', inplace=True)
#     predicted_sales.index.name = 'Date'

#     for date, row in predicted_sales.iterrows():
#         if date in complete_df[(complete_df['Store'] == i)].index:
#             complete_df.loc[(complete_df['Store'] == i) & (complete_df.index == date), 'Sales'] = row['TimeGPT']


## Check the updated data
# for i in missing_list_1:
#     df_check_period = complete_df.loc[(complete_df['Store'] == i) & (complete_df.index >= start_missing_date) & (complete_df.index <= end_missing_date)]

#     ## Ensure the data is complete for the forecasting
#     print(df_check_period)
#     print(complete_df)            
###-------------------------------------------------------------------------------
    



#%%#
### Step 3-2  ######################################################################
### Check the complete data set after imputation


## Overview of the data
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train_complete.info())
print("------------------------------------------------ \n")
print("\n • Processed training Data Set \n")
print(df_train_complete.head())
print("------------------------------------------------")
print("\n • Processed testing Data Set \n")
print(df_test_processed.head())

#%%#
### Step 3-3  ######################################################################
### Naive Approach for clustering 


## Filtering function for individual store
def filter_training_data(df, store_number):
    filtered_df = df[df['Store'] == store_number]
    return filtered_df


## Clustering: Assortment type and overall sales values
## Calculate the number of store that is belong to assortment type a, b, and c
assortment_type_a = []
assortment_type_b = []
assortment_type_c = []

## In df_store, if the df_store['Assortment'] = a, then append the corresponding values of store into list
for i in range(1, 1116):
    store_data = filter_training_data(df_store, i)
    if store_data['Assortment'].values == 'a':
        assortment_type_a.append(i)
    elif store_data['Assortment'].values == 'b':
        assortment_type_b.append(i)
    else:
        assortment_type_c.append(i)

## For store type a, b, and c respectively
store_type_a = []
store_type_b = []
store_type_c = []
store_type_d = []

## In df_store, if the df_store['StoreType'] = a, then append the corresponding values of store into list
for i in range(1, 1116):
    store_data = filter_training_data(df_store, i)
    if store_data['StoreType'].values == 'a':
        store_type_a.append(i)
    elif store_data['StoreType'].values == 'b':
        store_type_b.append(i)
    elif store_data['StoreType'].values == 'c':
        store_type_c.append(i)
    else:
        store_type_d.append(i)


###-------------------------------------------------------------------------------
        


#%%#
## Naive Approach1: Separating the store by assortment type, using daily sales data
## CLustering the store based on the assortment type, if the assortment type is a then using blue line, if the assortment type is b then using red line, c using green line
## Plotting sales pattern over the entire time cross all stores


## For assortment type a, b, and c respectively
plt.figure(figsize=(40, 7))
for i in assortment_type_a:
    store_data = filter_training_data(df_train_complete, i)
    plt.plot(store_data.index, store_data['Sales'], color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type A sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
# Plot sales for assortment type B
plt.figure(figsize=(40, 7))
for i in assortment_type_b:
    store_data = filter_training_data(df_train_complete, i)
    plt.plot(store_data.index, store_data['Sales'], color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type B sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
# Plot sales for assortment type C
plt.figure(figsize=(40, 7))
for i in assortment_type_c:
    store_data = filter_training_data(df_train_complete, i)
    plt.plot(store_data.index, store_data['Sales'], color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type C sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

###-------------------------------------------------------------------------------



#%%#
## Naive Approach2: Separating the store by assortment type, using weekly sales data
## CLustering the store based on the assortment type, if the assortment type is a then using blue line, if the assortment type is b then using red line, c using green line
## Plotting sales pattern over the entire time cross all stores


## For assortment type a, b, and c respectively
plt.figure(figsize=(40, 7))
for i in assortment_type_a:
    store_data = filter_training_data(df_train_complete, i)
    weekly_sales = store_data['Sales'].resample('W').sum()
    plt.plot(weekly_sales.index, weekly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type A weekly sales over time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()


## Plot sales for assortment type B
plt.figure(figsize=(40, 7))
for i in assortment_type_b:
    store_data = filter_training_data(df_train_complete, i)
    weekly_sales = store_data['Sales'].resample('W').sum()
    plt.plot(weekly_sales.index, weekly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type B weekly sales over time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()


## Plot sales for assortment type C
plt.figure(figsize=(40, 7))
for i in assortment_type_c:
    store_data = filter_training_data(df_train_complete, i)
    weekly_sales = store_data['Sales'].resample('W').sum()
    plt.plot(weekly_sales.index, weekly_sales, color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Assortment type C weekly sales over time')
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.show()


###-------------------------------------------------------------------------------




#%%#
## Naive Approach3: Separating the store by assortment type, using monthly sales data
## CLustering the store based on the assortment type, if the assortment type is a then using blue line, if the assortment type is b then using red line, c using green line
## Plotting sales pattern over the entire time cross all stores


## For assortment type a, b, and c respectively
plt.figure(figsize=(40, 7))
for i in assortment_type_a:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Assortment type A monthly sales over time')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

## Plot sales for assortment type B
plt.figure(figsize=(40, 7))
for i in assortment_type_b:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Assortment type B monthly sales over time')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

## Plot sales for assortment type C
plt.figure(figsize=(40, 7))
for i in assortment_type_c:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Assortment type C monthly sales over time')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


###-------------------------------------------------------------------------------



#%%#
## Naive Approach4: Separating the store by store type, using daily sales data
## CLustering the store based on the store type, if the store type is a then using blue line, if the store type is b then using red line, c using green line
## Plotting sales pattern over the entire time cross all stores


## For store type a, b, c, and d respectively
plt.figure(figsize=(40, 7))
for i in store_type_a:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Store type A sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

## Plot sales for store type B
plt.figure(figsize=(40, 7))
for i in store_type_b:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Store type B sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

## Plot sales for store type C
plt.figure(figsize=(40, 7))
for i in store_type_c:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Store type C sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


## Plot sales for store type D
plt.figure(figsize=(40, 7))
for i in store_type_d:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='darkorange')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.title('Store type D sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()


###-------------------------------------------------------------------------------



#%%#
## Naive Approach5: Separating the store by "have competition or not", using monthly sales data
## CLustering the store based on the competition, if the store has competition then using blue line, if the store has no competition then using red line
## Plotting sales pattern over the entire time cross all stores


## For store with competition and without competition respectively, but in same plot
plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['CompetitionOpenSinceMonth'].isnull().values[0]:
        plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
        ## Blue line for store without competition
    else:
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
        ## Red line for store with competition
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with and without competition')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


## Plot the sales pattern for stores with competition and without competition separately
plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['CompetitionOpenSinceMonth'].isnull().values[0]:
        plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores without competition')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if not df_store[df_store['Store'] == i]['CompetitionOpenSinceMonth'].isnull().values[0]:
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with competition')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()



###-------------------------------------------------------------------------------



#%%#
## Naive Approach6: Separating the store by those have competition, separated by the "starting year of competition", using monthly sales data
## CLustering the store based on the competition, if the store since before 2000 has competition then using blue line, and if the store between 2001 and 2007 has competition then using red line, and if the store since 2008 and 2015 has competition then using green line


## For store with from 2000, 2001-2007, and 2008-2015 respectively
store_competition_2000 = []
store_competition_2001_2007 = []
store_competition_2008_2015 = []

plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['CompetitionOpenSinceYear'].values[0] < 2000:
        store_competition_2000.append(i)
        plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
    elif 2000 <= df_store[df_store['Store'] == i]['CompetitionOpenSinceYear'].values[0] <= 2007:
        store_competition_2001_2007.append(i)
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
    else:
        store_competition_2008_2015.append(i)
        plt.plot(monthly_sales.index, monthly_sales, color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with competition since different years')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


## Plot the sales pattern for stores with competition since different years separately
plt.figure(figsize=(40, 7))
for i in store_competition_2000:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with competition since before 2000')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

plt.figure(figsize=(40, 7))
for i in store_competition_2001_2007:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with competition since 2001-2007')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

plt.figure(figsize=(40, 7))
for i in store_competition_2008_2015:
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='limegreen')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with competition since 2008-2015')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


###-------------------------------------------------------------------------------



#%%#
## Naive Approach7: Separating the store by have promotion or not, using monthly sales data
## CLustering the store based on the promotion, if the store has promotion then using blue line, and if the store has no promotion then using red line
## Plotting sales pattern over the entire time cross all stores


## For store with promotion and without promotion respectively
plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['Promo2'].values[0] == 0:
        plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
        ## Blue line for store without promotion!
    else:
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
        ## Red line for store with promotion!
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with and without promotion')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

## Plot the sales pattern for stores with promotion and without promotion separately
plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['Promo2'].values[0] == 0:
        plt.plot(monthly_sales.index, monthly_sales, color='royalblue')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores without promotion')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()

plt.figure(figsize=(40, 7))
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    if df_store[df_store['Store'] == i]['Promo2'].values[0] == 1:
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with promotion')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-4  ######################################################################
### Time Series Approach1: Sales data for all stores


## Cluster the stores based on the overall sales values, and set up several thresholds to define the clusters, for the clustering rule, we keep each cluster with ame amount of stores in it
## Calculate the average sales values for each store
store_sales = df_train_complete.groupby('Store')['Sales'].mean()


## Define the thresholds for clustering, assume there are 5 groups, this is just simply split the stores into 5 groups based on the sales values
thresholds = np.linspace(store_sales.min(), store_sales.max(), 6)


## Cluster the stores based on the average sales values plotting them in different colors
plt.figure(figsize=(40, 7))
colors = ['royalblue', 'lightcoral', 'limegreen', 'darkkhaki', 'darkorange']
for i in range(1, 1116):
    store_data = filter_training_data(df_train_complete, i)
    monthly_sales = store_data['Sales'].resample('M').sum()
    plt.plot(monthly_sales.index, monthly_sales, color='gray')
    if store_sales[i] < thresholds[-1]:  # Check against the last threshold
        for j in range(len(thresholds) - 1):  # Loop through all thresholds except the last one
            if store_sales[i] < thresholds[j]:
                plt.plot(monthly_sales.index, monthly_sales, color=colors[j])
                break
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with different sales values')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


###-------------------------------------------------------------------------------




#%%#
### Step 3-5  ######################################################################
### Time Series Approach2: Sales data for all stores using K-means
### K-means: Aim to cluster several groups of stores based on the exact amounts and timings of their sales, whereas K shape is more focused on the shapes of the sales patterns without considering the exact amounts and timings scales

## Clustering the stores based on the raw monthly sales values (without transformation)
## Define the number of clusters (groups), and calculate monthly average sales for each store, at this stage the data frame will be row=monthly sales vales start from 01-2013 to 07-2015, column=store number
## The determination of the number of clusters is based on the results of silhouette score calculation

num_clusters = 2
weekly_avg_sales = df_train_complete.groupby('Store')['Sales'].resample('M').mean().unstack(level=0)
## In the process of clustering, we current consider the weekly sales data, and we can also consider other external factors such as the store type, assortment type, competition, promotion, etc. that can reflect the features of the stores

## Transpose the data so that stores are rows and months are columns that can be processed by KMeans
weekly_avg_sales = weekly_avg_sales.T


## Define the K-means algorithm, and fit the data
## In the calculation behind K-means, the algorithm will calculate the distance between each store and the cluster center, and then assign the store to the cluster with the closest center
## And for the representation of each store will be a multidimensional vector, and the dimension is the number of weeks in the data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(weekly_avg_sales)
cluster_labels = kmeans.labels_
weekly_avg_sales['Cluster'] = cluster_labels


## Plot the clustering results by plotting the average sales for each cluster sequentially
plt.figure(figsize=(40, 7))
for cluster in range(num_clusters):
    cluster_data = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].drop(columns='Cluster')
    plt.plot(cluster_data.columns, cluster_data.mean(axis=0), label=f'Cluster {cluster}')
plt.title('Weekly Average Sales by Cluster')
plt.xlabel('Week')
plt.ylabel('Average Sales')
plt.legend()
plt.show()


## Plot the each store and the cluster centres in a 2 dimensional space
pca = PCA(n_components=2)
weekly_avg_sales_pca = pca.fit_transform(weekly_avg_sales.drop(columns='Cluster'))
plt.figure(figsize=(10, 10))
for cluster in range(num_clusters):
    cluster_data = weekly_avg_sales_pca[weekly_avg_sales['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', label='Cluster Center')
plt.title('PCA into two dims of Weekly Average Sales by Cluster')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


## Print the number of store in each cluster
print(weekly_avg_sales['Cluster'].value_counts())


## Print the store number in each cluster
for cluster in range(num_clusters):
    print(f"Cluster_{cluster+1}: {weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].index.tolist()}")


## Create multiple different list fot the storage the store number into each list for each cluster
Cluster_1_kmeans = []
Cluster_2_kmeans = []
Cluster_3_kmeans = []
Cluster_4_kmeans = []

for cluster in range(num_clusters):
    if cluster == 0:
        Cluster_1_kmeans = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].index.tolist()
    elif cluster == 1:
        Cluster_2_kmeans = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].index.tolist()
    elif cluster == 2:
        Cluster_3_kmeans = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].index.tolist()
    else:
        Cluster_4_means = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].index.tolist()



###-------------------------------------------------------------------------------



#%%#
### 3-5-1 ######################################################################
### Measure the similarity between the stores in the same cluster by calculating the correlation between the sales values of the stores in the same cluster
    

## Calculate the correlation between the sales values of the stores in the same cluster
average_correlation = []
for cluster in range(num_clusters):
    cluster_data = weekly_avg_sales[weekly_avg_sales['Cluster'] == cluster].drop(columns='Cluster')
    cluster_correlation = cluster_data.T.corr().values
    cluster_correlation = cluster_correlation[np.triu_indices(cluster_correlation.shape[0], k=1)]
    average_correlation.append(cluster_correlation.mean())
    print("------------------------------------------------")
    print(f"Cluster {cluster+1} Average Correlation: {cluster_correlation.mean()}")
print("------------------------------------------------ \n")
print(f"Overall Average Correlation: {np.mean(average_correlation)}")


## Calculate the silhouette score for the clustering results
silhouette = silhouette_score(weekly_avg_sales.drop(columns='Cluster'), cluster_labels)
print("------------------------------------------------ \n")
print(f"Silhouette Score for {num_clusters} clusters: {silhouette} \n")
print("------------------------------------------------")


###-------------------------------------------------------------------------------



#%%#
### Elbow Method and Silhouette Score for the optimal number of clusters


## Plot the silhouette score for different number of clusters
score_n_1 = [0]  ## Indicate the cluster=1's silhouette score=0
silhouette_scores = []
for k in range(2, 16):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(weekly_avg_sales.drop(columns='Cluster'))
    silhouette_avg = silhouette_score(weekly_avg_sales.drop(columns='Cluster'), cluster_labels)
    silhouette_scores.append(silhouette_avg)
all_scores = score_n_1 + silhouette_scores
plt.figure(figsize=(20, 5))
plt.plot(range(1, 16), all_scores, marker='o')
plt.axvline(x = all_scores.index(max(silhouette_scores)) +1, color='black', linestyle='--')
plt.title('Silhouette Score by Number of Clusters for monthly sales data')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(1, 16))
plt.xlim(0, 16)
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Silhouette Score for the optimal number of clusters


## Plot the Elbow Method to determine the optimal number of clusters
distortions = []
for k in range(1, 16):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(weekly_avg_sales.drop(columns='Cluster'))
    distortions.append(kmeans.inertia_)
plt.figure(figsize=(20, 5))
plt.plot(range(1, 16), distortions, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters for monthly sales data')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.xticks(range(1, 16))
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-6  ######################################################################
### Time Series Approach3: Sales data for all stores using KShape

## Clustering the stores based on the Kshape
## Build the store information extractor, and convert the ['Sales'] column into Integer type
# def filter_training_data(df, store_number):
#     filtered_df = df[df['Store'] == store_number]
#     filtered_df['Sales'] = filtered_df['Sales'].astype(int)
#     return filtered_df

# ## Build the store information extractor, especially for the monthly sales data for each store
# def filter_training_data_monthly(df, store_number):
#     filtered_df = df[df['Store'] == store_number]
#     filtered_df = filtered_df.resample('M').mean()
#     filtered_df['Sales'] = filtered_df['Sales'].astype(int)
#     return filtered_df

# ## For the purpose of clustering the time series data based on the sales values, use filter_training_data function to append the dataframes of each store into a list
# store_data_list_daily = []
# for i in range(1, 1116):
#     store_data = filter_training_data(df_train_complete, i)
#     store_data_list_daily.append(store_data)

# store_data_list_monthly = []
# for i in range(1, 1116):
#     store_data = filter_training_data_monthly(df_train_complete, i)
#     store_data_list_monthly.append(store_data)

# ## Define the time series converter function
# def align_timeseries_dataset(dfs, target_col='Sales'):
#     ## dfs should be a list of dataframes that contain the time series data
#     target_col = 'Sales'
#     ## Load dataframes, turns them into a time-series array, and stores them in a list
#     tsdata = []
#     for i, df in enumerate(dfs):
#         tsdata.append(df[target_col].values.tolist()[:])
#         ## Check the maximum length of each time series data
#         len_max = 0
#         for ts in tsdata:
#             if len(ts) > len_max:
#                 len_max = len(ts)
#         ## Assign the last data to align the length of the time series data
#         for i, ts in enumerate(tsdata):
#             len_add = len_max - len(ts)
#             tsdata[i] = ts + [ts[-1]] * len_add
    
#     tsdata = np.array(tsdata)
#     return tsdata


# def transform_timeseries_vectors(timeseries_dataset):
#     ## Transform vectors
#     stack_list = []
#     for j in range(len(timeseries_dataset)):
#         data = np.array(timeseries_dataset[j])
#         data = data.reshape((1, len(data))).T
#         stack_list.append(data)
#     ## Convert to one-dimensional array
#     transformed_data = np.stack(stack_list, axis=0)
#     return transformed_data


# ## Align the time series data and assign the target column to the 'Sales' column
# target_col = 'Sales'
# dfs = align_timeseries_dataset(store_data_list_monthly, target_col=target_col)
# transformed_data = transform_timeseries_vectors(dfs)

# ## Define the KShape model
# ## To calculate the cross-correlation, it must be normalized.
# ## TimeSeriesScalerMeanVariance will be the class that normalizes the data.
# stack_data = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(transformed_data)

# ## Setting KShape class, number of clusters, and random seed to ensure the reproducibility
# n_clusters = 15 
# seed = 299192
# np.random.seed(seed)
# ks = KShape(n_clusters=n_clusters, n_init=20, verbose=True, random_state=seed)
# y_pred = ks.fit_predict(stack_data)


# ## Plot n number of clusters for each cluster in different plots
# for yi in range(n_clusters):  
#     plt.figure(figsize=(40, 7))  
#     for xx in stack_data[y_pred == yi]:
#         plt.plot(xx.ravel(), "k-", alpha=.2)
#     if hasattr(ks, 'cluster_centers_'):  
#         plt.plot(ks.cluster_centers_[yi].ravel(), "r-")  
#     plt.title("Cluster %d" % (yi + 1))
#     plt.tight_layout()
#     plt.show()


# ## PLot all cluster centers in one plot and illustrate the differences between them
# plt.figure(figsize=(40, 7))
# colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))  
# for yi in range(n_clusters):
#     plt.plot(ks.cluster_centers_[yi].ravel(), label=f"Cluster {yi + 1}", color=colors[yi])
# plt.title("Comparison of Cluster Centers")
# plt.legend()
# plt.tight_layout()
# plt.show()


# ## Create 4 different list fot the storage the store number into each list for each cluster
# Cluster_1 = []
# Cluster_2 = []
# Cluster_3 = []
# Cluster_4 = []

# for i in range(1, 1116):
#     if y_pred[i-1] == 0:
#         Cluster_1.append(i)
#     elif y_pred[i-1] == 1:
#         Cluster_2.append(i)
#     elif y_pred[i-1] == 2:
#         Cluster_3.append(i)
#     else:
#         Cluster_4.append(i)


# ## List the number of stores in each cluster
# print("Number of stores in each cluster:")
# print(pd.Series(y_pred).value_counts())


###-------------------------------------------------------------------------------



#%%#
### Step 3-7  ######################################################################
### Machine Learning Modelling for for current stage, only consider the stores' sales patterns(time-seriesish data) for the futher forecasting


## Aggregating the sales data from the same cluster, there are 4 cluster in total
## For those store in same cluster, aggregate the store based dataframe into one training data but only select the 


## Determine the start and end date for the training data, the lenth of the training data is 2 years
## In the training data, we aim to train our model to learn the slaes pattern for each store in order to make a 48 days prediction in a one go
## To achieve that purpose, we determine the define the input data = 2013-01-01 to 2015-06-13, which lasts the 894 days
start_date = '2013-01-01'
end_date = '2015-06-13'
pred_start_date = '2015-06-14'
pred_end_date = '2015-07-31'


## Generate a dataframe of the store in CLuster_1, use pd.concat to concatenate the dataframes of the stores in the same cluster
cluster_1_body_list = []
for i in Cluster_1_kmeans:
    filtered_df = filter_training_data(df_train_complete, i)
    df = filtered_df.loc[(filtered_df.index >= start_date) & (filtered_df.index <= end_date)]
    df_pivot = df.pivot_table(index='Store', columns='Date_Column', values='Sales', aggfunc='sum')  
    cluster_1_body_list.append(df_pivot)

cluster_1_body_df = pd.concat(cluster_1_body_list, axis=0)


## Determine the validation data set, the length of the validation data is 47 days align with the test data date length
cluster_1_ans_list = []
for i in Cluster_1_kmeans:
    filtered_df = filter_training_data(df_train_complete, i)
    df = filtered_df.loc[(filtered_df.index >= pred_start_date) & (filtered_df.index <= pred_end_date)]
    df_pivot = df.pivot_table(index='Store', columns='Date_Column', values='Sales', aggfunc='sum')  
    cluster_1_ans_list.append(df_pivot)

cluster_1_ans_df = pd.concat(cluster_1_ans_list, axis=0)

## Entire training data set for cluster 1
cluster_1_train = pd.concat([cluster_1_body_df, cluster_1_ans_df], axis=1)


## Print the precessed body data set(2 year) for cluster 1
print("------------------------------------------------")
print("\n • Overview of the input for Cluster 1 \n")               
print(cluster_1_body_df.head())
print(len(cluster_1_body_df.columns))
print("------------------------------------------------")
print("\n • Overview of the output (Label) for Cluster 1 \n")
print(len(cluster_1_ans_df.columns))
print(cluster_1_ans_df.head())

print("------------------------------------------------")
print("\n • Overview of the entire data for Cluster 1 \n")
print(cluster_1_train.head())
print("------------------------------------------------")
print("\n • Shape of the entire data for Cluster 1 \n")
print(cluster_1_train.shape)


## Define the function to calculate the RMSPE
def calculate_rmspe(y_true, y_pred):
    epsilon = 1e-20  ## Small constant to avoid division by zero, add the small constant to the actual values to avoid division by zero 
    y_true = np.array(y_true, dtype=np.float32) ## Before calculating the RMSPE, bear in mind that the validation data set need to convert into numpy array
    y_pred = np.array(y_pred, dtype=np.float32)
    ## Calculating percentage errors only where actual values are non-zero
    mask = y_true != 0
    percentage_errors = np.square((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))
    rmspe = np.sqrt(np.mean(percentage_errors)) * 100
    return rmspe


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-1  ######################################################################
### Generate the training data set for the cluster 2 and the entire data that conbine cluster1 and cluster2


## Generate a dataframe of the store in CLuster_2, use pd.concat to concatenate the dataframes of the stores in the same cluster
cluster_2_body_list = []
for i in Cluster_2_kmeans:
    filtered_df = filter_training_data(df_train_complete, i)
    df = filtered_df.loc[(filtered_df.index >= start_date) & (filtered_df.index <= end_date)]
    df_pivot = df.pivot_table(index='Store', columns='Date_Column', values='Sales', aggfunc='sum')  
    cluster_2_body_list.append(df_pivot)

cluster_2_body_df = pd.concat(cluster_2_body_list, axis=0)


## Determine the validation data set, the length of the validation data is 47 days align with the test data date length
cluster_2_ans_list = []
for i in Cluster_2_kmeans:
    filtered_df = filter_training_data(df_train_complete, i)
    df = filtered_df.loc[(filtered_df.index >= pred_start_date) & (filtered_df.index <= pred_end_date)]
    df_pivot = df.pivot_table(index='Store', columns='Date_Column', values='Sales', aggfunc='sum')  
    cluster_2_ans_list.append(df_pivot)

cluster_2_ans_df = pd.concat(cluster_2_ans_list, axis=0)

## Entire training data set for cluster 2
cluster_2_train = pd.concat([cluster_2_body_df, cluster_2_ans_df], axis=1)


## Print the precessed body data set(2 year) for cluster 2
print("------------------------------------------------")
print("\n • Overview of the input for Cluster 2 \n")
print(cluster_2_body_df.head())
print(len(cluster_2_body_df.columns))
print("------------------------------------------------")
print("\n • Overview of the output (Label) for Cluster 2 \n")
print(len(cluster_2_ans_df.columns))
print(cluster_2_ans_df.head())
print("------------------------------------------------")
print("\n • Overview of the entire data for Cluster 2 \n")
print(cluster_2_train.head())
print("------------------------------------------------")
print("\n • Shape of the entire data for Cluster 2 \n")
print(cluster_2_train.shape)


## Combine the cluster 1 and cluster 2 data set
cluster_1_2_train = pd.concat([cluster_1_train, cluster_2_train], axis=0)
print("------------------------------------------------")
print("\n • Overview of the entire data for Cluster 1 and Cluster 2 \n")
print(cluster_1_2_train.head())
print("------------------------------------------------")
print("\n • Shape of the entire data for Cluster 1 and Cluster 2 \n")
print(cluster_1_2_train.shape)


###-------------------------------------------------------------------------------




#%%#
### Step 3-7-XGB-1  ######################################################################
### Build the XGBoost model for the cluster 1


## Define the model for the cluster 1
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores

# ## Define the training and validation set
# X_TRA = cluster_1_train.iloc[:685, :894] ## Shape (685, 894)
# Y_TRA = cluster_1_train.iloc[:685, 894:] ## Shape (685, 48)
# X_VLI = cluster_1_train.iloc[685:, :894] ## Shape (171, 894)
# Y_VLI = cluster_1_train.iloc[685:, 894:] ## Shape (171, 48)
# np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Tushar justification version
X_TRA = cluster_1_train.iloc[:685, :846] ## Shape (685, 846)
Y_TRA = cluster_1_train.iloc[:685, 846:894] ## Shape (685, 48)
X_VLI = cluster_1_train.iloc[685:, :846] ## Shape (171, 846)
Y_VLI = cluster_1_train.iloc[685:, 846:894] ## Shape (171, 48)
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_VLI.shape, Y_VLI.shape)


## Try XGBoost model for the cluster 1
train = xgb.DMatrix(X_TRA, Y_TRA)
test = xgb.DMatrix(X_VLI, Y_VLI)


## Define the parameters for the XGBoost model
param = {
        'max_depth': 500, 
        'learning_rate': 0.01, 
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse'}
num_round = 1000
## Train the model
model = xgb.train(param, train, num_round)
## Predict the validation set
y_pred = model.predict(test)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)



## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-XGB-1&2  ######################################################################
### Build the XGBoost model for the entire clusters 1 and 2


## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_2_train) * 0.8) ## =892 stores

## Define the training and validation set
X_TRA = cluster_1_2_train.iloc[:892, :894] 
Y_TRA = cluster_1_2_train.iloc[:892, 894:] 
X_VLI = cluster_1_2_train.iloc[892:, :894] 
Y_VLI = cluster_1_2_train.iloc[892:, 894:]
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 and Cluster 2 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 and Cluster 2 \n")
print(X_VLI.shape, Y_VLI.shape)


## Try XGBoost model for the cluster 1 and cluster 2
train = xgb.DMatrix(X_TRA, Y_TRA)
test = xgb.DMatrix(X_VLI, Y_VLI)


## Define the parameters for the XGBoost model
param = {
        'max_depth': 100, 
        'learning_rate': 0.1, 
        'objective': 'reg:squarederror', 
        'eval_metric': 'rmse'
        }
## Leaarning rate is aim to control the step size in updating the weights, the smaller the learning rate, the slower the model learns, and the better the model generalizes
## However, the larger the learning rate, the faster the model learns, but the model may not generalize well, which could lead to overfitting
num_round = 1000
## Number of rounds ia aim to control the number of boosting rounds, the more rounds, the more complex the model, and the more likely the model will overfit

## Train the model
model = xgb.train(param, train, num_round)
## Predict the validation set
y_pred = model.predict(test)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for Cluster 1 and Cluster 2 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")

## Visualize the prediction and the actual sales values for the first store in the cluster 1 and cluster 2's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1 and Cluster 2')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------


#%%#
### Step 3-7-XGB-CV  ######################################################################
### Cross-validation for the cluster 1 under XGB model


## Define the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch, including all evaluation metrics such as MSE, RMSE, MAE, MAPE, R2, and RMSPE
def cross_validation_cluster_1(X, Y):
    ## Define the parameters for the XGBoost model
    param = {
            'max_depth': 100, 
            'learning_rate': 0.01, 
            'objective': 'reg:squarederror', 
            'eval_metric': 'rmse'
            }
    num_round = 1000
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmspes_socre = []
    for train_index, test_index in kf.split(X):
        ## Data set splitting
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        Y_train, Y_test = Y.iloc[train_index, :], Y.iloc[test_index, :]
        np_Y_test = Y_test.to_numpy()
        # ## Model building
        # train = xgb.DMatrix(X_train, Y_train) ## XGBoost need sepcial data format, DMatrix
        test = xgb.DMatrix(X_test, Y_test)
        # model = xgb.train(param, train, num_round)
        y_pred = model.predict(test)
        ## Evaluation scores calculation
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, y_pred)
        mape = mean_absolute_percentage_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        rmspe_list = []
        for i in range(len(y_pred)):
            rmspe = calculate_rmspe(np_Y_test[i], y_pred[i])
            rmspe_list.append(rmspe)
        rmspe = np.mean(rmspe_list)
        ## Overall score lists
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        rmspes_socre.append(rmspe)

    return mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre


## Illustrate the different fold of the cross-validation scores for the cluster 1, for example, check the first fold's index and the second fold's index to see the difference
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(X_TRA)):
    print(f"Fold {i + 1}")
    print(f"Train index: {train_index}")
    print(f"Validation index: {test_index}")
    print("------------------------------------------------")


## Illustrate the cross-validation scores for the cluster 1 and 2 or entire data set
mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1(X_TRA, Y_TRA)


## Print the cross-validation scores for the cluster 1 and 2 or entire data set
print("------------------------------------------------")
print("\n • Cross-validation scores for the Cluster 1 \n")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MAPE: {np.mean(mape_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Mean RMSPE: {np.mean(rmspes_socre)}")


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-XGB-CV-Opt  ######################################################################
## Define the list of the parameters for the XGBoost model that after cross-validation to find the optimal parameters
# param_list = [{'max_depth': 5, 'eta': 0.1, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
#               {'max_depth': 10, 'eta': 0.1, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
#               {'max_depth': 15, 'eta': 0.1, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'},
#               {'max_depth': 20, 'eta': 0.1, 'objective': 'reg:squarederror', 'eval_metric': 'rmse'}]


# ## Define the list of the number of rounds for the XGBoost model that after cross-validation to find the optimal number of rounds
# num_round_list = [100, 500, 600, 800, 1000]


# ## Define the optimal parameters and the number of rounds
# param = {}
# num_round = 0
# best_rmse = float('inf')


# ## Loop through the parameters and the number of rounds to find the optimal parameters and the number of rounds
# for p in param_list:
#     for n in num_round_list:
#         param['max_depth'] = p['max_depth']
#         param['eta'] = p['eta']
#         param['objective'] = p['objective']
#         param['eval_metric'] = p['eval_metric']
#         num_round = n
#         mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores = cross_validation_cluster_1(X_TRA, Y_TRA)
#         if np.mean(rmse_scores) < best_rmse:
#             best_rmse = np.mean(rmse_scores)
#             param['max_depth'] = p['max_depth']
#             param['eta'] = p['eta']
#             param['objective'] = p['objective']
#             param['eval_metric'] = p['eval_metric']
#             num_round = n


# ## Illustrate the optimal parameters for the XGBoost model after cross-validation
# print("------------------------------------------------")
# print("\n • Optimal parameters for the XGBoost model after cross-validation \n")
# print(f"Optimal max_depth: {param['max_depth']}")
# print(f"Optimal eta: {param['eta']}")
# print(f"Optimal objective: {param['objective']}")
# print(f"Optimal eval_metric: {param['eval_metric']}")
# print(f"Optimal num_round: {num_round}")


## The Optimal parameter for the XGBoost model after cross-validation
##



###-------------------------------------------------------------------------------



#%%#
### Step 3-7-RF  ######################################################################
### Random Forest model for the cluster 1


## Define the Random Forest model for the cluster 1
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial


## Define the training and validation set
X_TRA = cluster_1_train.iloc[:685, :894] 
Y_TRA = cluster_1_train.iloc[:685, 894:] 
X_VLI = cluster_1_train.iloc[685:, :894]
Y_VLI = cluster_1_train.iloc[685:, 894:] 
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_VLI.shape, Y_VLI.shape)


## Try Random Forest model for the cluster 1
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_TRA, Y_TRA)
y_pred = model.predict(X_VLI)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)

## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-RF-CV  ######################################################################
### Cross-validation for the cluster 1 under Random Forest model

## Define the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch
def cross_validation_cluster_1_rf(X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmspes_socre = []
    for train_index, test_index in kf.split(X):
        ## Data set splitting
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        np_Y_test = Y_test.to_numpy()
        # ## Model building and training
        # model = RandomForestRegressor(n_estimators=1000, random_state=42)
        # model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        ## Evaluation metrics calculation
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, y_pred)
        mape = mean_absolute_percentage_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        rmspe_list = []
        for i in range(len(y_pred)):
            rmspe = calculate_rmspe(np_Y_test[i], y_pred[i])
            rmspe_list.append(rmspe)
        rmspe = np.mean(rmspe_list)
        ## Overall score lists
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        rmspes_socre.append(rmspe)

    return mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre


## Illustrate the cross-validation scores for the cluster 1
mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_rf(X_TRA, Y_TRA)
print("------------------------------------------------")
print("\n • Cross-validation scores for the Cluster 1 (average mse, rmse, and etc.) for Random Forest Model \n")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MAPE: {np.mean(mape_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Mean RMSPE: {np.mean(rmspes_socre)}")


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-RF-CV-Opt  ######################################################################
## Define the list of the parameters for the Random Forest model that after cross-validation to find the optimal parameters
param_list = [{'n_estimators': 100, 'random_state': 42},
                {'n_estimators': 250, 'random_state': 42},
                {'n_estimators': 500, 'random_state': 42},
                {'n_estimators': 750, 'random_state': 42},
                {'n_estimators': 1000, 'random_state': 42},
                {'n_estimators': 1500, 'random_state': 42},
                {'n_estimators': 2000, 'random_state': 42}]
## Define the optimal parameters and the number of rounds
param = {}
best_rmse = float('inf')


## Loop through the parameters and the number of rounds to find the optimal parameters and the number of rounds
for p in param_list:
    model = RandomForestRegressor(n_estimators=p['n_estimators'], random_state=p['random_state'])
    mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_rf(X_TRA, Y_TRA)
    if np.mean(rmse_scores) < best_rmse:
        best_rmse = np.mean(rmse_scores)
        param['n_estimators'] = p['n_estimators']
        param['random_state'] = p['random_state']


## Illustrate the optimal parameters for the Random Forest model after cross-validation
print("------------------------------------------------")
print("\n • Optimal parameters for the Random Forest model after cross-validation \n")
print(f"Optimal n_estimators: {param['n_estimators']}")
print("------------------------------------------------")
print(f"Optimal R2: {np.mean(r2_scores)}")
print(f"Optimal RMSPE: {np.mean(rmspes_socre)}")


## The Optimal parameter for the Random Forest model after cross-validation
## n_estimators: 100


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN  ######################################################################
### Neural Network model for the cluster 1

## Define the Neural Network model for the cluster 1
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial


## Define the training and validation set
X_TRA = cluster_1_train.iloc[:685, :894]
Y_TRA = cluster_1_train.iloc[:685, 894:] 
X_VLI = cluster_1_train.iloc[685:, :894] 
Y_VLI = cluster_1_train.iloc[685:, 894:] 
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

# ## Tushar justification version
# X_TRA = cluster_1_train.iloc[:685, :846] ## Shape (685, 846)
# Y_TRA = cluster_1_train.iloc[:685, 846:894] ## Shape (685, 48)
# X_VLI = cluster_1_train.iloc[685:, :846] ## Shape (171, 846)
# Y_VLI = cluster_1_train.iloc[685:, 846:894] ## Shape (171, 48)
# np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

# X_TEST = cluster_1_train.iloc[:, 48:894] 
# Y_TEST = cluster_1_train.iloc[:, 894:]

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_VLI.shape, Y_VLI.shape)
print("\n")


## Try Sequential Neural Network model for the cluster 1, using Keras sequential model
## For the general usage of different activation functions such as Sigmod, Tanh, ReLU, etc., will be briefly description in the following comments

## Sigmiod: Mainly use in the input=any real world vlaue, output=0 to 1, for an exaample, the output is the probability of the class like spam or not spam email

## Tahn: Mainly use in the input=any real world vlaue, output=-1 to 1, for an example, the output will potentially be the thing will have three class, like emotion, happy, sad, and neutral classification

## ReLU: Mainly use in the senario of the intput=any real world vlaue, output=0 to infinity, for an example, the output will be the thing that is non-negative, like the sales data, demand data, etc.

## Softmax: Mainly use in the senario of the input=any real world value, output=0 to 1, for an example, the output will be the thing that is the probability of the class, like the classification problem, but! the sum of the output will be 1, so this is more suitable for multi-class classification problem

## Linear: Mainly use in the senario of the input=any real world value, output=any real world value, for an example, the output will be the thing that is the regression problem, like the sales data, demand data, etc.

## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative
model = Sequential()
model.add(Dense(2048, input_dim=894, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='linear')) 
## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

# optimizer = Adam(learning_rate=0.001) 
model.compile(loss='mean_squared_error', optimizer='adam')
## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

model.fit(X_TRA, Y_TRA, epochs=1200, batch_size=64, verbose=0) 
## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## Predict the validation set, and I f the number of batch size is too small, the model will not be able to learn the pattern of the sales data, if the number of batch size is too large, the model will be overfitting

y_pred = model.predict(X_VLI)



## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-Tushar  ######################################################################
### Neural Network model for the cluster 1


# ## Define the training and validation set
# X_TRA = cluster_1_train.iloc[:685, :894]
# Y_TRA = cluster_1_train.iloc[:685, 894:] 
# X_VLI = cluster_1_train.iloc[685:, :894] 
# Y_VLI = cluster_1_train.iloc[685:, 894:] 
# np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Tushar justification version
X_TRA = cluster_1_train.iloc[:685, :846] ## Shape (685, 846)
Y_TRA = cluster_1_train.iloc[:685, 846:894] ## Shape (685, 48)
X_VLI = cluster_1_train.iloc[685:, :846] ## Shape (171, 846)
Y_VLI = cluster_1_train.iloc[685:, 846:894] ## Shape (171, 48)
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

XX_TRA = pd.concat([X_TRA, X_VLI], axis=0)
YY_TRA = pd.concat([Y_TRA, Y_VLI], axis=0)

X_TEST = cluster_1_train.iloc[:, 48:894]
Y_TEST = cluster_1_train.iloc[:, 894:]
np_Y_TEST = Y_TEST.to_numpy()

## Convert the columns from date to normal integer, for XGBoost model only since the model does not accept the different columns name that has difference between the training and the testing data set
XX_TRA.columns = range(XX_TRA.shape[1])
YY_TRA.columns = range(YY_TRA.shape[1])
X_TEST.columns = range(X_TEST.shape[1])
Y_TEST.columns = range(Y_TEST.shape[1])

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(XX_TRA.shape, YY_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_TEST.shape, Y_TEST.shape)
print("\n")


# ## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative
# model = Sequential()
# model.add(Dense(2048, input_dim=846, activation='relu')) ##Input dimension is 730(730 days)
# model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(48, activation='linear')) 
# ## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

# # optimizer = Adam(learning_rate=0.001) 
# model.compile(loss='mean_squared_error', optimizer='adam')
# ## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

# model.fit(XX_TRA, YY_TRA, epochs=1200, batch_size=64, verbose=0)
# ## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
# ## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
# y_pred = model.predict(X_TEST)


# ## Build a xgboost for the traditional approach
# ## Using the converted data set in case of the input data set has different columns name
# train = xgb.DMatrix(XX_TRA, YY_TRA)
# # test = xgb.DMatrix(X_VLI, Y_VLI)

# param = {
#         'max_depth': 100, 
#         'learning_rate': 0.1, 
#         'objective': 'reg:squarederror', 
#         'eval_metric': 'rmse'
#         }
# num_round = 1000

# model = xgb.train(param, train, num_round) 

# ## Predict the test set
# real_test = xgb.DMatrix(X_TEST)
# y_pred = model.predict(real_test)



## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_TEST, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_TEST, y_pred)
mape = mean_absolute_percentage_error(Y_TEST, y_pred)
r2 = r2_score(Y_TEST, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_TEST[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-Tushar-2  ######################################################################
### Neural Network model for the cluster 1 Time sliceing


## Drop the last 48 columns first
DF_total = cluster_1_train.iloc[:, :-48]

## Ensure we have the right number of columns
num_columns = DF_total.shape[1]
input_days = 798  
forecast_days = 48


## Proceed with data slicing
num_rows = len(DF_total)
X_TRA = []
Y_TRA = []

for i in range(num_rows):
    max_start = num_columns - input_days - forecast_days
    for start_day in range(max_start + 1):  # Ensuring we have at least one possible start
        end_day = start_day + input_days
        forecast_end_day = end_day + forecast_days
        input_data = DF_total.iloc[i, start_day:end_day]
        output_data = DF_total.iloc[i, end_day:forecast_end_day]
        X_TRA.append(input_data.values)
        Y_TRA.append(output_data.values)

## Convert to DataFrames and then to Numpy if required for further processing
X_TRA_array = np.array(X_TRA)
Y_TRA_array = np.array(Y_TRA)
lag_body = pd.DataFrame(X_TRA_array)
lag_answer = pd.DataFrame(Y_TRA_array)

lag_total = pd.concat([lag_body, lag_answer], axis=1)

num_training = int(0.8 * lag_total.shape[0])
X_TRA = lag_total.iloc[:num_training, :input_days]
Y_TRA = lag_total.iloc[:num_training, input_days:]
X_VLI = lag_total.iloc[num_training:, :input_days]
Y_VLI = lag_total.iloc[num_training:, input_days:]

XX_TRA = pd.concat([X_TRA, X_VLI], axis=0)
YY_TRA = pd.concat([Y_TRA, Y_VLI], axis=0)

## Real answer
y_test = cluster_1_train.iloc[:, 846:894]
x_test = cluster_1_train.iloc[:, 48:846]
np_test = y_test.to_numpy()

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data + validation data set for Cluster 1 \n")
print(XX_TRA.shape, YY_TRA.shape)



## Try Sequential Neural Network model for the cluster 1, using Keras sequential model
## For the general usage of different activation functions such as Sigmod, Tanh, ReLU, etc., will be briefly description in the following comments

## Sigmiod: Mainly use in the input=any real world vlaue, output=0 to 1, for an exaample, the output is the probability of the class like spam or not spam email

## Tahn: Mainly use in the input=any real world vlaue, output=-1 to 1, for an example, the output will potentially be the thing will have three class, like emotion, happy, sad, and neutral classification

## ReLU: Mainly use in the senario of the intput=any real world vlaue, output=0 to infinity, for an example, the output will be the thing that is non-negative, like the sales data, demand data, etc.

## Softmax: Mainly use in the senario of the input=any real world value, output=0 to 1, for an example, the output will be the thing that is the probability of the class, like the classification problem, but! the sum of the output will be 1, so this is more suitable for multi-class classification problem

## Linear: Mainly use in the senario of the input=any real world value, output=any real world value, for an example, the output will be the thing that is the regression problem, like the sales data, demand data, etc.

## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative

model = Sequential()
model.add(Dense(2048, input_dim=798, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='linear')) 

## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

# optimizer = Adam(learning_rate=0.001) 

model.compile(loss='mean_squared_error', optimizer='adam')

## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

model.fit(XX_TRA, YY_TRA, epochs=1200, batch_size=64, verbose=0) 

## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## Predict the validation set, and I f the number of batch size is too small, the model will not be able to learn the pattern of the sales data, if the number of batch size is too large, the model will be overfitting

y_pred = model.predict(x_test)


# ## Build a XGBoost for the time slicing approach
# train = xgb.DMatrix(XX_TRA, YY_TRA)
# # test = xgb.DMatrix(X_VLI, Y_VLI)
# real_test = xgb.DMatrix(x_test)
# param = {
#         'max_depth': 100,
#         'learning_rate': 0.1,
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse'
#         }
# num_round = 1000
# model = xgb.train(param, train, num_round)

# ## Predict the test set
# y_pred = model.predict(real_test)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_test[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(y_test.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()




###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-slicing  ######################################################################
### Neural Network model for the cluster 1 with time slicing data set


## Define the Neural Network model for the cluster 1
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial


# ## Define the training and validation set
# X_TRA = cluster_1_train.iloc[:685, :894]
# Y_TRA = cluster_1_train.iloc[:685, 894:] 
# X_VLI = cluster_1_train.iloc[685:, :894] 
# Y_VLI = cluster_1_train.iloc[685:, 894:] 
# np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
# print("------------------------------------------------")
# print("\n • Shape of the training data set for Cluster 1 \n")
# print(X_TRA.shape, Y_TRA.shape)
# print("------------------------------------------------")
# print("\n • Shape of the validation data set for Cluster 1 \n")
# print(X_VLI.shape, Y_VLI.shape)
# print("\n")


## Time Slicing the data set, for the purpose of learn the sales patter start from any time point, which lasts two years(two seasonalit)
## Rebuild the data set by using the time slicing format, in the original data each row contain 894 days standing for the body part, and 48 days standing for the answer part, but now we brshape the row one into only 730 days start from 2013-01-01 to 2014-12-31, and the answer part is the 48 days start from 2015-01-01 to 2015-02-17 for store 1, and the next row will be the 730 days start from 2013-01-02 to 2015-01-01, and the answer part is the 48 days start from 2015-01-02 to 2015-02-18 for store 1, and so on


## Assuming cluster_1_train is your DataFrame and is already defined
num_rows = len(cluster_1_train)
input_days = 730
forecast_days = 48

## Initialize lists to store input and output data
X_TRA = []
Y_TRA = []

## Iterate over each store
for i in range(num_rows):
    # Ensure there are enough days for the last input + forecast
    max_start = cluster_1_train.shape[1] - input_days - forecast_days
    for start_day in range(max_start + 1):  # +1 to include the last possible start
        end_day = start_day + input_days
        forecast_end_day = end_day + forecast_days
        # Extract input and output data
        input_data = cluster_1_train.iloc[i, start_day:end_day]
        output_data = cluster_1_train.iloc[i, end_day:forecast_end_day]
        X_TRA.append(input_data.values)
        Y_TRA.append(output_data.values)

## Convert lists to numpy arrays if needed for further processing with machine learning models
X_TRA_lag = np.array(X_TRA)  ## [141240 rows x 730 columns]
Y_TRA_lag = np.array(Y_TRA)
lag_body = pd.DataFrame(X_TRA_lag)  ## the lag body part under the first cluster
lag_answer = pd.DataFrame(Y_TRA_lag)  ## the lag answer part under the first cluster

lag_total = pd.concat([lag_body, lag_answer], axis=1)  

## Split the lag_total data set into training and validation set following the 80-20 rule
X_TRA = lag_total.iloc[:113000, :730] ## Shape (113000, 730) 34188
Y_TRA = lag_total.iloc[:113000, 730:] ## Shape (113000, 48)
X_VLI = lag_total.iloc[113000:, :730] ## Shape (28260, 730)
Y_VLI = lag_total.iloc[113000:, 730:] ## Shape (28260, 48)
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

XX_TRA = pd.concat([X_TRA, X_VLI], axis=0)
YY_TRA = pd.concat([Y_TRA, Y_VLI], axis=0)

## Illustrat the shape of the training data and the validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of entire data set for Cluster 1 \n")
print(XX_TRA.shape, YY_TRA.shape)
print("------------------------------------------------")

# print("------------------------------------------------")
# print("\n • Shape of the validation data set for Cluster 1 \n")
# print(X_VLI.shape, Y_VLI.shape)
# print("\n")


## Try Sequential Neural Network model for the cluster 1, using Keras sequential model
## For the general usage of different activation functions such as Sigmod, Tanh, ReLU, etc., will be briefly description in the following comments

## Sigmiod: Mainly use in the input=any real world vlaue, output=0 to 1, for an exaample, the output is the probability of the class like spam or not spam email

## Tahn: Mainly use in the input=any real world vlaue, output=-1 to 1, for an example, the output will potentially be the thing will have three class, like emotion, happy, sad, and neutral classification

## ReLU: Mainly use in the senario of the intput=any real world vlaue, output=0 to infinity, for an example, the output will be the thing that is non-negative, like the sales data, demand data, etc.

## Softmax: Mainly use in the senario of the input=any real world value, output=0 to 1, for an example, the output will be the thing that is the probability of the class, like the classification problem, but! the sum of the output will be 1, so this is more suitable for multi-class classification problem

## Linear: Mainly use in the senario of the input=any real world value, output=any real world value, for an example, the output will be the thing that is the regression problem, like the sales data, demand data, etc.

## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative
model = Sequential()
model.add(Dense(2048, input_dim=730, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='linear')) 
## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

# optimizer = Adam(learning_rate=0.001) 
model.compile(loss='mean_squared_error', optimizer='adam')
## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

model.fit(X_TRA, Y_TRA, epochs=800, batch_size=64, verbose=0) 
## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## Predict the validation set, and I f the number of batch size is too small, the model will not be able to learn the pattern of the sales data, if the number of batch size is too large, the model will be overfitting

y_pred = model.predict(X_VLI)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# ## save model 
# filepath = './{Model Name}'
# save_model(model, filepath, save_format='h5')

# ## load model
# test_load_model = load_model(
#     filepath = './{Model Name}',
#     custom_objects=None,
#     compile=True
# )


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-CV-slicing  ######################################################################
### Cross-validation for the cluster 1 under Neural Network model

## Define the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch, ensuring the model is not overfitting,  if the CV score is as good as the training score, the model is not overfitting


## Definne the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch by using the previous trained model without training the model again



def cross_validation_cluster_1_nn(X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmspes_socre = []
    for train_index, test_index in kf.split(X):
        ## Data set splitting
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        np_Y_test = Y_test.to_numpy()
        ## Model building
        model = Sequential()
        model.add(Dense(2048, input_dim=730, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(48, activation='linear'))
        # optimizer = Adam(learning_rate=0.01)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=800, batch_size=64, verbose=0)
        y_pred = model.predict(X_test) ### Change the model Name
        ## Evaluation metrics calculating
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, y_pred)
        mape = mean_absolute_percentage_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        rmspe_list = []
        for i in range(len(y_pred)):
            rmspe = calculate_rmspe(np_Y_test[i], y_pred[i])
            rmspe_list.append(rmspe)
        rmspe = np.mean(rmspe_list)
        ## Overall score lists
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        rmspes_socre.append(rmspe)
    return mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre


## Illustrate the cross-validation scores for the cluster 1, 
## Make sure using the XX_TRA and YY_TRA data set, which is the entire data set for the cluster 1
mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_nn(XX_TRA, YY_TRA)


print("------------------------------------------------")
print("\n • Cross-validation scores for the Cluster 1 (average mse, rmse, and etc.) \n")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MAPE: {np.mean(mape_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Mean RMSPE: {np.mean(rmspes_socre)}%")


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-Model-train  ######################################################################
### Neural Network model for the cluster 1 with the entire data set

## For cluster 1
## Assuming cluster_1_train is your DataFrame and is already defined
num_rows = len(cluster_1_train)
input_days = 730
forecast_days = 48

## Initialize lists to store input and output data
X_TRA = []
Y_TRA = []

## Iterate over each store
for i in range(num_rows):
    # Ensure there are enough days for the last input + forecast
    max_start = cluster_1_train.shape[1] - input_days - forecast_days
    for start_day in range(max_start + 1):  # +1 to include the last possible start
        end_day = start_day + input_days
        forecast_end_day = end_day + forecast_days
        # Extract input and output data
        input_data = cluster_1_train.iloc[i, start_day:end_day]
        output_data = cluster_1_train.iloc[i, end_day:forecast_end_day]
        X_TRA.append(input_data.values)
        Y_TRA.append(output_data.values)

## Convert lists to numpy arrays if needed for further processing with machine learning models
X_TRA_lag = np.array(X_TRA)  ## [141240 rows x 730 columns]
Y_TRA_lag = np.array(Y_TRA)
lag_body = pd.DataFrame(X_TRA_lag)  ## the lag body part under the first cluster
lag_answer = pd.DataFrame(Y_TRA_lag)  ## the lag answer part under the first cluster

lag_total = pd.concat([lag_body, lag_answer], axis=1)  

## Split the lag_total data set into training and validation set following the 80-20 rule
X_TRA = lag_total.iloc[:113000, :730] ## Shape (113000, 730) 34188
Y_TRA = lag_total.iloc[:113000, 730:] ## Shape (113000, 48)
X_VLI = lag_total.iloc[113000:, :730] ## Shape (28260, 730)
Y_VLI = lag_total.iloc[113000:, 730:] ## Shape (28260, 48)
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

XX_TRA = pd.concat([X_TRA, X_VLI], axis=0)
YY_TRA = pd.concat([Y_TRA, Y_VLI], axis=0)

## Illustrat the shape of the training data and the validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of entire data set for Cluster 1 \n")
print(XX_TRA.shape, YY_TRA.shape)
print("------------------------------------------------")

## Build neural network model for the cluster 1
model = Sequential()
model.add(Dense(2048, input_dim=730, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='linear')) 
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(XX_TRA, YY_TRA, epochs=800, batch_size=64, verbose=0)

## Save the cluster 1 model 
filepath = './cluster_1_nn'
save_model(model, filepath, save_format='h5')


###-------------------------------------------------------------------------------




#%%#
### Step 3-7-NN-Model-train2  ######################################################################
### Neural Network model for the cluster 2 with the entire data set


## For cluster 1
## Assuming cluster_1_train is your DataFrame and is already defined
num_rows = len(cluster_2_train)
input_days = 730
forecast_days = 48

## Initialize lists to store input and output data
X_TRA = []
Y_TRA = []

## Iterate over each store
for i in range(num_rows):
    # Ensure there are enough days for the last input + forecast
    max_start = cluster_2_train.shape[1] - input_days - forecast_days
    for start_day in range(max_start + 1):  # +1 to include the last possible start
        end_day = start_day + input_days
        forecast_end_day = end_day + forecast_days
        # Extract input and output data
        input_data = cluster_2_train.iloc[i, start_day:end_day]
        output_data = cluster_2_train.iloc[i, end_day:forecast_end_day]
        X_TRA.append(input_data.values)
        Y_TRA.append(output_data.values)

## Convert lists to numpy arrays if needed for further processing with machine learning models
X_TRA_lag = np.array(X_TRA)  ## [141240 rows x 730 columns]
Y_TRA_lag = np.array(Y_TRA)
lag_body = pd.DataFrame(X_TRA_lag)  ## the lag body part under the first cluster
lag_answer = pd.DataFrame(Y_TRA_lag)  ## the lag answer part under the first cluster

lag_total = pd.concat([lag_body, lag_answer], axis=1)  

## Split the lag_total data set into training and validation set following the 80-20 rule
X_TRA = lag_total.iloc[:34188, :730] ## Shape (113000, 730) 34188
Y_TRA = lag_total.iloc[:34188, 730:] ## Shape (113000, 48)
X_VLI = lag_total.iloc[34188:, :730] ## Shape (28260, 730)
Y_VLI = lag_total.iloc[34188:, 730:] ## Shape (28260, 48)
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

XX_TRA = pd.concat([X_TRA, X_VLI], axis=0)
YY_TRA = pd.concat([Y_TRA, Y_VLI], axis=0)

## Illustrat the shape of the training data and the validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of entire data set for Cluster 1 \n")
print(XX_TRA.shape, YY_TRA.shape)
print("------------------------------------------------")

## Build neural network model for the cluster 1
model = Sequential()
model.add(Dense(2048, input_dim=730, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='linear')) 
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(XX_TRA, YY_TRA, epochs=800, batch_size=64, verbose=0)

## Save the cluster 1 model 
filepath = './cluster_2_nn'
save_model(model, filepath, save_format='h5')


###-------------------------------------------------------------------------------


#%%#
### step 3-7-NN-Opt  ######################################################################
## Define the list of the parameters for the Neural Network model that after cross-validation to find the optimal parameters, there are 8 layers
param_list = [
    {'layer1': 1024, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 1024, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 1024, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 1024, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 2048, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 800},
    {'layer1': 2048, 'layer2': 512, 'layer3': 512, 'layer4': 256, 'layer5': 256, 'layer7':128, 'layer8':64, 'epochs': 800},
            ]
## Define the optimal parameters and the number of rounds
param = {}
num_round = 0
best_rmse = float('inf')

## Loop through the parameters and the number of rounds to find the optimal parameters and the number of rounds
for p in param_list:
    param['layer1'] = p['layer1']
    param['layer2'] = p['layer2']
    param['layer3'] = p['layer3']
    param['layer4'] = p['layer4']
    param['layer5'] = p['layer5']
    param['layer7'] = p['layer7']
    param['layer8'] = p['layer8']
    param['epochs'] = p['epochs']
    mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_nn(X_TRA, Y_TRA)
    if np.mean(rmse_scores) < best_rmse:
        best_rmse = np.mean(rmse_scores)
        param['layer1'] = p['layer1']
        param['layer2'] = p['layer2']
        param['layer3'] = p['layer3']
        param['layer4'] = p['layer4']
        param['layer5'] = p['layer5']
        param['layer7'] = p['layer7']
        param['layer8'] = p['layer8']
        param['epochs'] = p['epochs']

## Illustrate the optimal parameters for the Neural Network model after cross-validation
print("------------------------------------------------")
print("\n • Optimal parameters for the Neural Network model after cross-validation \n")
print(f"Optimal layer1: {param['layer1']}")
print(f"Optimal layer2: {param['layer2']}")
print(f"Optimal layer3: {param['layer3']}")
print(f"Optimal layers: {param['layer4']}")
print(f"Optimal layers: {param['layer5']}")
print(f"Optimal layers: {param['layer7']}")
print(f"Optimal layers: {param['layer8']}")
print(f"Optimal epochs: {param['epochs']}")
print(f"Optimal num_round: {num_round}")
print("------------------------------------------------")
print(f"Optimal R2: {np.mean(r2_scores)}")
print(f"Optimal RMSPE: {np.mean(rmspes_socre)}")

## Teh optimal parameters fot the Neural Network model after cross-validation
## layer1: 1024, layer2: 512, layer3: 128, epochs: 1000, num_round: 0


###-------------------------------------------------------------------------------



#%%#
#%%#
### Step 3-7-NN-nor  ######################################################################
### Neural Network model for the cluster 1 with the normalized data set


## Define the Neural Network model for the cluster 1 with the normalized data set
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial

## Normilax=ze the training data set
scaler = MinMaxScaler()
cluster_1_train = pd.DataFrame(scaler.fit_transform(cluster_1_train), columns=cluster_1_train.columns)


## Define the training and validation set
X_TRA_nor = cluster_1_train.iloc[:685, :894]
Y_TRA_nor = cluster_1_train.iloc[:685, 894:] 
X_VLI_nor = cluster_1_train.iloc[685:, :894] 
Y_VLI_nor = cluster_1_train.iloc[685:, 894:] 
np_Y_VLI_nor = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA_nor.shape, Y_TRA_nor.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_VLI_nor.shape, Y_VLI_nor.shape)
print("\n")


## Try Sequential Neural Network model for the cluster 1, using Keras sequential model
## For the general usage of different activation functions such as Sigmod, Tanh, ReLU, etc., will be briefly description in the following comments

## Sigmod: Mainly use in the input=any real world vlaue, output=0 to 1, for an exaample, the output is the probability of the class like spam or not spam email

## Tahn: Mainly use in the input=any real world vlaue, output=-1 to 1, for an example, the output will potentially be the thing will have three class, like emotion, happy, sad, and neutral classification

## ReLU: Mainly use in the senario of the intput=any real world vlaue, output=0 to infinity, for an example, the output will be the thing that is non-negative, like the sales data, demand data, etc.

## Softmax: Mainly use in the senario of the input=any real world value, output=0 to 1, for an example, the output will be the thing that is the probability of the class, like the classification problem, but! the sum of the output will be 1, so this is more suitable for multi-class classification problem

## Linear: Mainly use in the senario of the input=any real world value, output=any real world value, for an example, the output will be the thing that is the regression problem, like the sales data, demand data, etc.

## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative
model = Sequential()
model.add(Dense(2048, input_dim=894, activation='relu')) ##Input dimension is 730(730 days)
model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='sigmoid')) 
## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

optimizer = Adam(learning_rate=0.00001)
## In terms of the learning rate, the learning rate is the most important hyperparameter in the deep learning field, the learning rate is the rate at which the model will learn the pattern of the data, if the learning rate is too small, the model will not be able to learn the pattern of the data, if the learning rate is too large, the model will be overfitting

model.compile(loss='mean_squared_error', optimizer=optimizer)
## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

model.fit(X_TRA_nor, Y_TRA_nor, epochs=1200, batch_size=64, verbose=0) 
## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
## Predict the validation set, and I f the number of batch size is too small, the model will not be able to learn the pattern of the sales data, if the number of batch size is too large, the model will be overfitting

y_pred_nor = model.predict(X_VLI_nor)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI_nor, y_pred_nor)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI_nor, y_pred_nor)
mape = mean_absolute_percentage_error(Y_VLI_nor, y_pred_nor)
r2 = r2_score(Y_VLI_nor, y_pred_nor)
rmspes = []
for i in range(len(y_pred_nor)):
    rmspe = calculate_rmspe(np_Y_VLI_nor[i], y_pred_nor[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI_nor.iloc[0, :].values, label='Actual')
plt.plot(y_pred_nor[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-NN-CV-nor  ######################################################################
### Cross-validation for the cluster 1 under Neural Network model with the normalized data set


## Define the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch, ensuring the model is not overfitting,  if the CV score is as good as the training score, the model is not overfitting
def cross_validation_cluster_1_nn(X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmspes_socre = []
    for train_index, test_index in kf.split(X):
        ## Data set splitting
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        np_Y_test = Y_test.to_numpy()
        ## Model building
        model = Sequential()
        model.add(Dense(2048, input_dim=894, activation='relu'))
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(48, activation='sigmoid'))
        optimizer = Adam(learning_rate=0.00001)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        model.fit(X_train, Y_train, epochs=800, batch_size=64, verbose=0)
        y_pred = model.predict(X_test)
        ## Evaluation metrics calculating
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, y_pred)
        mape = mean_absolute_percentage_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        rmspe_list = []
        for i in range(len(y_pred)):
            rmspe = calculate_rmspe(np_Y_test[i], y_pred[i])
            rmspe_list.append(rmspe)
        rmspe = np.mean(rmspe_list)
        ## Overall score lists
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        rmspes_socre.append(rmspe)
    return mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre


## Illustrate the cross-validation scores for the cluster 1
mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_nn(X_TRA_nor, Y_TRA_nor)

print("------------------------------------------------")
print("\n • Cross-validation scores for the Cluster 1 (average mse, rmse, and etc.) \n")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MAPE: {np.mean(mape_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Mean RMSPE: {np.mean(rmspes_socre)}")


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-LSTM  ######################################################################
### Buliding the LSTM model for the cluster 1

## Define the LSTM model for the cluster 1 
## Split the data into training and validation set following the 80-20 rule
training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial

## Define the training and validation set
X_TRA = cluster_1_train.iloc[:685, :894]
Y_TRA = cluster_1_train.iloc[:685, 894:]
X_VLI = cluster_1_train.iloc[685:, :894]
Y_VLI = cluster_1_train.iloc[685:, 894:]
np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE

## Illustrate the shape and the format of the training data set and validation data set
print("------------------------------------------------")
print("\n • Shape of the training data set for Cluster 1 \n")
print(X_TRA.shape, Y_TRA.shape)
print("------------------------------------------------")
print("\n • Shape of the validation data set for Cluster 1 \n")
print(X_VLI.shape, Y_VLI.shape)
print("\n")


## Try LSTM model for the cluster 1, using Keras sequential model
model = Sequential()
model.add(LSTM(2048, input_shape=(894, 1), return_sequences=True))
model.add(LSTM(1024, return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(48))
model.add(Dense(48, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_TRA, Y_TRA, epochs=1200, batch_size=64, verbose=0)
y_pred = model.predict(X_VLI)


## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
mse = mean_squared_error(Y_VLI, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_VLI, y_pred)
mape = mean_absolute_percentage_error(Y_VLI, y_pred)
r2 = r2_score(Y_VLI, y_pred)
rmspes = []
for i in range(len(y_pred)):
    rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
    rmspes.append(rmspe)
overall_rmspe = np.mean(rmspes)


## Print the evaluation scores
print("------------------------------------------------")
print("\n • Evaluation scores for the first store under Cluster 1 \n")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"R2: {r2}")
print(f"RMSPE: {overall_rmspe}%")
print("------------------------------------------------")


## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
plt.figure(figsize=(40, 7))
plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
plt.plot(y_pred[0], label='Predicted')
plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-7-LSTM-CV  ######################################################################
### Cross-validation for the cluster 1 under LSTM model

## Define the cross-validation function for the cluster 1 to see the original parameters configuration's performance on the cluster 1 by using the cross-validation appraoch, ensuring the model is not overfitting,  if the CV score is as good as the training score, the model is not overfitting
def cross_validation_cluster_1_lstm(X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    r2_scores = []
    rmspes_socre = []
    for train_index, test_index in kf.split(X):
        ## Data set splitting
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        np_Y_test = Y_test.to_numpy()
        ## Model building
        model = Sequential()
        model.add(LSTM(2048, input_shape=(894, 1), return_sequences=True))
        model.add(LSTM(1024, return_sequences=True))
        model.add(LSTM(512, return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64, return_sequences=True))
        model.add(LSTM(48))
        model.add(Dense(48, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=1200, batch_size=64, verbose=0)
        y_pred = model.predict(X_test)
        ## Evaluation metrics calculating
        mse = mean_squared_error(Y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Y_test, y_pred)
        mape = mean_absolute_percentage_error(Y_test, y_pred)
        r2 = r2_score(Y_test, y_pred)
        rmspe_list = []
        for i in range(len(y_pred)):
            rmspe = calculate_rmspe(np_Y_test[i], y_pred[i])
            rmspe_list.append(rmspe)
        rmspe = np.mean(rmspe_list)
        ## Overall score lists
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
        r2_scores.append(r2)
        rmspes_socre.append(rmspe)
    return mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre


## Illustrate the cross-validation scores for the cluster 1
mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_lstm(X_TRA, Y_TRA)
print("------------------------------------------------")
print("\n • Cross-validation scores for the Cluster 1 (average mse, rmse, and etc.) \n")
print(f"Mean MSE: {np.mean(mse_scores)}")
print(f"Mean RMSE: {np.mean(rmse_scores)}")
print(f"Mean MAE: {np.mean(mae_scores)}")
print(f"Mean MAPE: {np.mean(mape_scores)}")
print(f"Mean R2: {np.mean(r2_scores)}")
print(f"Mean RMSPE: {np.mean(rmspes_socre)}")


###-------------------------------------------------------------------------------



#%%#
# ### Step 3-7-NN-slicing  ######################################################################
# ### Neural Network model for the cluster 1

# ## Define the Neural Network model for the cluster 1
# ## Split the data into training and validation set following the 80-20 rule
# training_number = int(len(cluster_1_train) * 0.8) ## =685 stores, but this is just a trial


# ## Define the training and validation set
# X_TRA = cluster_1_train.iloc[:685, :894]
# Y_TRA = cluster_1_train.iloc[:685, 894:] 
# X_VLI = cluster_1_train.iloc[685:, :894] 
# Y_VLI = cluster_1_train.iloc[685:, 894:] 
# np_Y_VLI = Y_VLI.to_numpy() ## For calculating the RMSPE
# ## In the training set, the body part is 894 days, and the target is 48 days, Hwever in this case, we try to reduce the body part into 800 days and the target is 48 days still the same, the main differnece is that original start date is 2013-01-01 to 2015-06-13 as the training data set and the 2015-06-14 to 2015-07-31 as the validation data set, but in this case, we create a data of 730 day and the target is right after the end date, in terms of this, the first row will be 2013-01-01 to 2015-01-01 and the validatoi data set will be 2015-01-02 to 2015-02-19 under store 1, and the next row will be 2013-01-02 to 2015-01-02 and the validation data set will be 2015-01-03 to 2015-02-20 under store 1 as well, till the store 1's validation data end at 2015-07-31, then move on the next store.

# ## Run a loop to slice the training and validation data set form the original data set start form the row one
# X_TRA_sliced = []
# Y_TRA_sliced = []
# X_VLI_sliced = []
# Y_VLI_sliced = []
# for i in range(685):
#     X_TRA_sliced.append(X_TRA.iloc[i:i+730, :])
#     Y_TRA_sliced.append(Y_TRA.iloc[i, :])
#     X_VLI_sliced.append(X_VLI.iloc[i:i+730, :])
#     Y_VLI_sliced.append(Y_VLI.iloc[i, :])

# ## Convert the list into the numpy array
# X_TRA_sliced = np.array(X_TRA_sliced)
# Y_TRA_sliced = np.array(Y_TRA_sliced)
# X_VLI_sliced = np.array(X_VLI_sliced)
# Y_VLI_sliced = np.array(Y_VLI_sliced)


# ## Illustrate the shape and the format of the training data set and validation data set
# print("------------------------------------------------")
# print("\n • Shape of the training data set for Cluster 1 \n")
# print(X_TRA.shape, Y_TRA.shape)
# print("------------------------------------------------")
# print("\n • Shape of the validation data set for Cluster 1 \n")
# print(X_VLI.shape, Y_VLI.shape)
# print("\n")


# ## Try Sequential Neural Network model for the cluster 1, using Keras sequential model
# ## For the general usage of different activation functions such as Sigmod, Tanh, ReLU, etc., will be briefly description in the following comments

# ## Sigmod: Mainly use in the input=any real world vlaue, output=0 to 1, for an exaample, the output is the probability of the class like spam or not spam email

# ## Tahn: Mainly use in the input=any real world vlaue, output=-1 to 1, for an example, the output will potentially be the thing will have three class, like emotion, happy, sad, and neutral classification

# ## ReLU: Mainly use in the senario of the intput=any real world vlaue, output=0 to infinity, for an example, the output will be the thing that is non-negative, like the sales data, demand data, etc.

# ## Softmax: Mainly use in the senario of the input=any real world value, output=0 to 1, for an example, the output will be the thing that is the probability of the class, like the classification problem, but! the sum of the output will be 1, so this is more suitable for multi-class classification problem

# ## Linear: Mainly use in the senario of the input=any real world value, output=any real world value, for an example, the output will be the thing that is the regression problem, like the sales data, demand data, etc.

# ## Interms of the usage of the activation='relu' is due to the reason thay the sales data is non-negative
# model = Sequential()
# model.add(Dense(2048, input_dim=894, activation='relu')) ##Input dimension is 730(730 days)
# model.add(Dense(1024, activation='relu')) ## ReLU is more accurate to capture the non-negative sales
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(48, activation='linear')) 
# ## Output dimension is 48(48 days), the selection of linear activation function is that, luinear is more accurate to capture the slaes number=0 than ReLU, if there is a subtle variation in the sales number around 0, the linear activation function is more accurate to capture the subtle variation 

# # optimizer = Adam(learning_rate=0.001) 
# model.compile(loss='mean_squared_error', optimizer='adam')
# ## The main reason to use the mean squared error is that the sales data is continuous, and the mean squared error is more accurate to capture the continuous data, the optimizer is adam, which is the most popular optimizer in the deep learning field

# model.fit(X_TRA, Y_TRA, epochs=1200, batch_size=64, verbose=0) 
# ## Epoch is mainly for the number of times the model will go through the entire training data set, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
# ## If the epoch is too small, the model will not be able to learn the pattern of the sales data, if the epoch is too large, the model will be overfitting, the batch size is the number of samples that will be used in each epoch, the verbose is the output of the training process
# ## Predict the validation set, and I f the number of batch size is too small, the model will not be able to learn the pattern of the sales data, if the number of batch size is too large, the model will be overfitting


# y_pred = model.predict(X_VLI)


# ## Calculate the MSE, RMSE, MAP, MAPE, R2 socres for the validation set
# mse = mean_squared_error(Y_VLI, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(Y_VLI, y_pred)
# mape = mean_absolute_percentage_error(Y_VLI, y_pred)
# r2 = r2_score(Y_VLI, y_pred)
# rmspes = []
# for i in range(len(y_pred)):
#     rmspe = calculate_rmspe(np_Y_VLI[i], y_pred[i])
#     rmspes.append(rmspe)
# overall_rmspe = np.mean(rmspes)


# ## Print the evaluation scores
# print("------------------------------------------------")
# print("\n • Evaluation scores for the first store under Cluster 1 \n")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")
# print(f"MAE: {mae}")
# print(f"MAPE: {mape}")
# print(f"R2: {r2}")
# print(f"RMSPE: {overall_rmspe}%")
# print("------------------------------------------------")


# ## Visualize the prediction and the actual sales values for the first store in the cluster 1's validation set
# plt.figure(figsize=(40, 7))
# plt.plot(Y_VLI.iloc[0, :].values, label='Actual')
# plt.plot(y_pred[0], label='Predicted')
# plt.title('Actual vs Predicted Sales for the first store under Cluster 1')
# plt.xlabel('Date')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()


# ###-------------------------------------------------------------------------------



#%%#
### Step 3-7-LSTM-Opt  ######################################################################
## Define the list of the parameters for the LSTM model that after cross-validation to find the optimal parameters, there are 8 layers
param_list = [
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
    {'layer1': 2048, 'layer2': 1024, 'layer3': 512, 'layer4': 256, 'layer5': 128, 'layer7':128, 'layer8':64, 'epochs': 1200},
            ]
## Define the optimal parameters and the number of rounds
param = {}
num_round = 0
best_rmse = float('inf')

## Loop through the parameters and the number of rounds to find the optimal parameters and the number of rounds
for p in param_list:
    param['layer1'] = p['layer1']
    param['layer2'] = p['layer2']
    param['layer3'] = p['layer3']
    param['layer4'] = p['layer4']
    param['layer5'] = p['layer5']
    param['layer7'] = p['layer7']
    param['layer8'] = p['layer8']
    param['epochs'] = p['epochs']
    mse_scores, rmse_scores, mae_scores, mape_scores, r2_scores, rmspes_socre = cross_validation_cluster_1_lstm(X_TRA, Y_TRA)
    if np.mean(rmse_scores) < best_rmse:
        best_rmse = np.mean(rmse_scores)
        param['layer1'] = p['layer1']
        param['layer2'] = p['layer2']
        param['layer3'] = p['layer3']
        param['layer4'] = p['layer4']
        param['layer5'] = p['layer5']
        param['layer7'] = p['layer7']
        param['layer8'] = p['layer8']
        param['epochs'] = p['epochs']

## Illustrate the optimal parameters for the LSTM model after cross-validation
print("------------------------------------------------")
print("\n • Optimal parameters for the LSTM model after cross-validation \n")
print(f"Optimal layer1: {param['layer1']}")
print(f"Optimal layer2: {param['layer2']}")
print(f"Optimal layer3: {param['layer3']}")
print(f"Optimal layers: {param['layer4']}")
print(f"Optimal layers: {param['layer5']}")
print(f"Optimal layers: {param['layer7']}")
print(f"Optimal layers: {param['layer8']}")
print(f"Optimal epochs: {param['epochs']}")
print(f"Optimal num_round: {num_round}")
print("------------------------------------------------")
print(f"Optimal R2: {np.mean(r2_scores)}")
print(f"Optimal RMSPE: {np.mean(rmspes_socre)}")

## The optimal parameters for the LSTM model after cross-validation
## layer1: 2048, layer2: 1024, layer3: 512, epochs: 1200, num_round: 0


###-------------------------------------------------------------------------------



#%%#
### Step Final Prediction ######################################################################


## Load the models for both cluster 1 and cluster 2
model_cluster_1 = load_model(
    filepath='/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/cluster_1_nn',
    custom_objects=None,
    compile=True
)

model_cluster_2 = load_model(
    filepath='/Users/ttonny0326/GitHub_Project/sales-forecasting-ml-models/cluster_2_nn',
    custom_objects=None,
    compile=True
)

## Check the information of the model
print("------------------------------------------------")
print("\n • Information of the model for Cluster 1 \n")
print(model_cluster_1.summary())
print("------------------------------------------------")
print("\n • Information of the model for Cluster 2 \n")
print(model_cluster_2.summary())
print("------------------------------------------------")



## Make the final forcasting for the cluster 1 and cluster 2
## Define the testing data set
x_test = cluster_1_2_train.iloc[:, 212:]

## If the store number is in Cluster_1_kmeans then use the model_cluster_1 to predict the sales, if the store number is in Cluster_2_kmeans then use the model_cluster_2 to predict the sales
## Using the index of x_test to filter the store number, the store umber should be 1 to 1115
store = []
for i in range(1, 1116):
    store.append(i)

y_pred = []
for i in store:
    if i in Cluster_1_kmeans:
        y_pred.append(model_cluster_1.predict(x_test.loc[i].values.reshape(1, -1)))
    else:
        y_pred.append(model_cluster_2.predict(x_test.loc[i].values.reshape(1, -1)))

## Convert the list into the data frame with the row stand for the store number and the column stand for the sales
y_pred = pd.DataFrame(np.array(y_pred).reshape(-1, 48))



## Illustrate the final prediction
print("------------------------------------------------")
print("\n • Final prediction for the sales \n")
print(y_pred)
print("------------------------------------------------")


## Conver the y_pred columns into the date format start from 2015-08-01 to 2015-09-17
date = pd.date_range(start='2015-08-01', end='2015-09-17')
y_pred.columns = date


## And replace all negative values with 0
y_pred[y_pred < 0] = 0


## Print aspecific store's prediction in plt
plt.figure(figsize=(40, 7))
plt.plot(y_pred.loc[1], label='Predicted')
plt.title('Predicted Sales for the store 2')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


## Plot the merge version of the previous sales data and the predicted sales data for store number one start form 2013-01-01 to 2015-09-17
plt.figure(figsize=(40, 7))
plt.plot(cluster_1_2_train.loc[1, :], label='Actual', color='royalblue')
plt.plot(y_pred.loc[1], label='Predicted', color='red')
plt.title('Entire Sales for the store 2')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()







 #%%#
### Step 4-1  ######################################################################
### Advanced data processed for store information 


store_info_data_cleaned = df_store.drop(df_store.columns[10], axis=1)
store_info_data_cleaned = store_info_data_cleaned.drop(store_info_data_cleaned.columns[10], axis=1)


## Loading the Store.csv
print("\n • Overview info of the store information \n")
print(store_info_data_cleaned.info())
print("------------------------------------------------")
print("\n • Overview info of Store CSV \n")
print(store_info_data_cleaned)
print("------------------------------------------------")
print("\n • Overview info of Store CSV's columns \n")
print(store_info_data_cleaned.columns)


## Since there are only 3 missing values out of 1115 in 'CompetitionDistance', we can fill them with the mean of the column
## And the distribution of 'CompetitionDistance' is right-skewed, so it is better to use median
## Checking for missing values in 'CompetitionDistance' and filling them with the mean of the column
plt.plot(store_info_data_cleaned['Store'],store_info_data_cleaned['CompetitionDistance'], label='Distance')
plt.title('Competition Distance for All Stores')
plt.xlabel('Store')
plt.ylabel('Distance')

comp_dist_median = store_info_data_cleaned['CompetitionDistance'].median()
store_info_data_cleaned['CompetitionDistance'].fillna(comp_dist_median, inplace=True)


## Convert categorical variables 'StoreType' and 'Assortment'
## For 'StoreType', we can use OneHotEncoding with 3 columns
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = encoder.fit_transform(store_info_data_cleaned[['StoreType']])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['StoreType']))
store_info_data_cleaned = store_info_data_cleaned.drop(['StoreType'], axis=1)
store_info_data_cleaned = pd.concat([store_info_data_cleaned, encoded_features_df], axis=1)


## For 'Assortment', we can use LabelEncoding
label_encoder = LabelEncoder()
store_info_data_cleaned['Assortment'] = label_encoder.fit_transform(store_info_data_cleaned['Assortment'])
store_info_data_cleaned['Assortment'] = store_info_data_cleaned['Assortment'] + 1


## Display the cleaned data
print("------------------------------------------------")
print("\n • Overview of the Store CSV \n")
print(store_info_data_cleaned.head())
print("------------------------------------------------")
print("\n • Overview of Store CSV's columns \n")
print(store_info_data_cleaned.columns.tolist())


###-------------------------------------------------------------------------------



#%%#
### Step 4-2  ######################################################################
## Handling missing values for 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear' using mode


# comp_month_mode = store_info_data_cleaned['CompetitionOpenSinceMonth'].mode()[0]
# comp_year_mode = store_info_data_cleaned['CompetitionOpenSinceYear'].mode()[0]
# store_info_data_cleaned['CompetitionOpenSinceMonth'].fillna(comp_month_mode, inplace=True)
# store_info_data_cleaned['CompetitionOpenSinceYear'].fillna(comp_year_mode, inplace=True)


# ## Handling missing values for 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval' using 0 and 'NotApplicable'
# store_info_data_cleaned['Promo2SinceWeek'].fillna(0, inplace=True)
# store_info_data_cleaned['Promo2SinceYear'].fillna(0, inplace=True)
# store_info_data_cleaned['PromoInterval'].fillna('NotApplicable', inplace=True)



# ## Display the cleaned data
# print("------------------------------------------------")
# print("\n • Overview of the Store CSV \n")
# print(store_info_data_cleaned.head())
# print("------------------------------------------------")
# print("\n • Overview of Store CSV's columns \n")
# print(store_info_data_cleaned.columns.tolist())
###-------------------------------------------------------------------------------

## Save the processed data
# store_info_data_cleaned.to_csv("processed_store_info_data_v1.csv")
###-------------------------------------------------------------------------------



#%%#
### Step 4-3  ######################################################################
### Advanced data processed for overall Store information in missing values


## Stats Approaches for the column have missing values
## For CompetitionOpenSinceMonth -> Distribution
## For CompetitionOpenSinceYear -> Distribution
## For Promo2SinceWeek -> Distribution
## For Promo2SinceYear -> Distribution

## Using other stats approaches to fill the missing values, such as mean, median, mode, etc.
## Handling missing values for 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear' using mode
store_info_stats = store_info_data_cleaned.copy()

comp_month_mode = store_info_stats['CompetitionOpenSinceMonth'].mode()[0]
comp_year_mode = store_info_stats['CompetitionOpenSinceYear'].mode()[0]

store_info_stats['CompetitionOpenSinceMonth'].fillna(comp_month_mode, inplace=True)
store_info_stats['CompetitionOpenSinceYear'].fillna(comp_year_mode, inplace=True)


## Handling missing values for 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval' using 0 and 'NotApplicable'
store_info_stats['Promo2SinceWeek'].fillna(0, inplace=True)
store_info_stats['Promo2SinceYear'].fillna(0, inplace=True)
store_info_stats['PromoInterval'].fillna('NotApplicable', inplace=True)


## Display the cleaned data
print("------------------------------------------------")
print("\n • Overview of the Store CSV - Stats\n")
print(store_info_stats.head())
print("------------------------------------------------")
print("\n • Info of the Store CSV - Stats\n")
print(store_info_stats.info())
print("------------------------------------------------")
print("\n • Overview of Store CSV's columns - Stats \n")
print(store_info_stats.columns.tolist())


###-------------------------------------------------------------------------------



#%%#
## Correlation checking for the Store csv file
store_regression_check = store_info_data_cleaned.copy()
checking_data = store_regression_check.dropna(subset=['CompetitionOpenSinceMonth'])  # , 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'
correlation_matrix = checking_data[['Assortment', 'CompetitionDistance', 'Promo2', 'StoreType_b', 'StoreType_c', 'StoreType_d', 'CompetitionOpenSinceMonth']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for prediction of CompetitionOpenSinceMonth')
plt.show()




#%%#
### Step 4-4  ######################################################################
## Using Regression to predict missing values


## Using Linear Regression to predict missing values for "CompetitionOpenSinceMonth"
store_regression = store_info_data_cleaned.copy()
columns_to_impute = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

for column in columns_to_impute:
    ## Train a model only on rows where the target column is not missing
    complete_data = store_regression.dropna(subset=[column])
    
    ## Observe the correlation between the column that dont have missing values
    ## Define features (X) and target (y)
    X = complete_data[['Assortment', 'CompetitionDistance', 'Promo2', 'StoreType_b', 'StoreType_c', 'StoreType_d']]
    y = complete_data[column]
    ## Initialize and train the regression model
    model = LinearRegression()
    model.fit(X, y)
    ## Predict missing values only
    missing_data_indices = store_regression[column].isnull()
    X_missing = store_regression.loc[missing_data_indices, ['Assortment', 'CompetitionDistance', 'Promo2', 'StoreType_b', 'StoreType_c', 'StoreType_d']]
    ## Ensure there are missing entries to impute
    if not X_missing.empty:
        predictions = model.predict(X_missing)
        ## If the column represents a year, we'll want to round to the nearest whole number
        store_regression.loc[missing_data_indices, column] = np.rint(predictions).astype(int)


## Display the DataFrame with imputed values
print("------------------------------------------------")
print("\n • Overview of the Store CSV - Regression for Numerical columns \n")
print(store_regression.head(20))



## Using RNN to predict missing values



## Aboves are the steps to process the Store.csv file, using predictive approaches to fill the missing values
###-------------------------------------------------------------------------------


