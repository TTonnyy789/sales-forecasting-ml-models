#%%#
### Step 1  ######################################################################
### Import essential library and load the data


## Import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import soft_dtw
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV, KFold, StratifiedKFold, cross_val_predict
from nixtlats import NixtlaClient
from nixtlats.date_features import CountryHolidays
from pytorch_forecasting import TimeSeriesDataSet


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

## In df_store, if the df_store['StoreType'] = a, then append the corresponding values of store into list
for i in range(1, 1116):
    store_data = filter_training_data(df_store, i)
    if store_data['StoreType'].values == 'a':
        store_type_a.append(i)
    elif store_data['StoreType'].values == 'b':
        store_type_b.append(i)
    else:
        store_type_c.append(i)


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
# Plot sales for assortment type B
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
# Plot sales for assortment type C
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
# Plot sales for assortment type B
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
# Plot sales for assortment type C
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

## For store type a, b, and c respectively
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
# Plot sales for store type B
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
# Plot sales for store type C
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
        ## Blue line for store without promotion
    else:
        plt.plot(monthly_sales.index, monthly_sales, color='lightcoral')
        ## Red line for store with promotion
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
        plt.plot(monthly_sales.index, monthly_sales, color='gray')
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
        plt.plot(monthly_sales.index, monthly_sales, color='darkkhaki')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Monthly sales over time for stores with promotion')
plt.xlabel('Date')
plt.ylabel('Monthly Sales')
plt.show()


###-------------------------------------------------------------------------------



#%%#
### Step 3-4  ######################################################################
### Time Series clustering: Sales data for all stores









#%%#
### Step 3-5  ######################################################################
### Temporal Clustering: Transformed sales data for all stores











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



## Using KNN to predict missing values



## Aboves are the steps to process the Store.csv file, using predictive approaches to fill the missing values
###-------------------------------------------------------------------------------


