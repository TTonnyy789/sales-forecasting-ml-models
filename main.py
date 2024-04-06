#%%#
### Step 1  ######################################################################
### Import essential library and load the data


## Import the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Read the data 
df_store = pd.read_csv('/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/DA2024_stores.csv')

df_train = pd.read_csv('/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/DA2024_train.csv')

df_test = pd.read_csv('/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/DA2024_test.csv')


## Overview of the data
print("\n • Overview info of the store information")
print(" ")
print(df_store.info())
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train.info())
print("------------------------------------------------")

print("\n • Training Data Set \n")
print(df_train.head())
print("------------------------------------------------")
print("\n • Testing Data Set \n")
print(df_test.head())
###-------------------------------------------------------------------------------



#%%#
### Step 2  ######################################################################
### Data Preprocessing


## Deal with the missing values
store_missing = df_store.isnull().sum()
train_missing = df_train.isnull().sum()


## Address the missing values with ... methods(mean value, median value, mode value, etc.)
## ...

## Convert datetime into correct datetime format also configure the index into the datetime
df_train['Date'] = pd.to_datetime(df_train['Date'], dayfirst=True)
df_train.set_index('Date', inplace=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], dayfirst=True)
df_test.set_index('Date', inplace=True)


## Check the updated training data and missing values among the store information and training data
print("\n • Training Data \n")
print(df_train.head())
print("------------------------------------------------")
print("\n • Testing Data \n")
print(df_test.head())
print("------------------------------------------------")
print("\n • Missing values in the store information")
print(store_missing)
print("------------------------------------------------")
print("\n • Missing values in the training data")
print(train_missing)
print("------------------------------------------------")


## Save the processed data
# df_train.to_csv("processed_training_data.csv")
# df_test.to_csv("processed_testing_data.csv")
###-------------------------------------------------------------------------------



#%%#
### Step 3  ######################################################################
### Data Exploration


## Loading the precessed data 
df_train_processed = pd.read_csv('/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/processed_training_data.csv')

df_test_processed = pd.read_csv('/Users/ttonny0326/BA_ORRA/Term 2/Data Analytics/Course_work/processed_testing_data.csv')


## Overview of the data
print("------------------------------------------------")
print("\n • Overview info of the training data \n")
print(df_train_processed.info())
print("------------------------------------------------")

print("\n • Processed training Data Set \n")
print(df_train.head())
print("------------------------------------------------")
print("\n • Processed testing Data Set \n")
print(df_test.head())
###-------------------------------------------------------------------------------



#%%#
### Step 4  ######################################################################
### Advanced data processed


## Group the data by store
store_grouped = df_train.groupby('Store')
store_grouped.head()

## Get store number 1's data
store_1 = store_grouped.get_group(1)
store_1.head()

## ...


## Setting the visualisation configurations for the plots
sns.set_theme()
sns.set_style("darkgrid")
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

overall_sales_1 = store_1.groupby('Date')['Sales'].sum()

## Creating the plots for basic visualisation 
plt.figure(figsize=(14, 7))
plt.plot(overall_sales_1, label='Sales')
plt.title('Sales over time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()


# %%
