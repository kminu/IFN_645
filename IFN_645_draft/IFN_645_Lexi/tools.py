# inside dm_tools.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

def data_prep():
    df = pd.read_csv('CaseStudy1-data/CaseStudyData.csv')
    
    # find the missing data for all features
    MissingData = df.isnull().sum()
    print(MissingData.sort_values(ascending = False))


    # there are 16 columns that are uniformly has missing data
    # drop the missing values of subset columns total 44 instances
    print('\n\n ********************Dropping the missing data***************************')
    df=df.dropna(subset=['PRIMEUNIT', 'AUCGUART','VehYear','Make','Color','Transmission','WheelTypeID','WarrantyCost', \
                   'VehOdo','Nationality','Size','TopThreeAmericanName','IsOnlineSale','VehBCost','VNST','Auction'])
    
    ## VehYear ##
    df['VehYear'] = pd.Categorical(df['VehYear'])
    
    ## COLOR ##
    # Replace '?' into 'SILVER'
    df['Color'] = df['Color'].replace('?', 'SILVER')
    
    ## TRANSMISSION ##
    #Replace ? => Auto
    #Replace Manual => MANUAL
    df['Transmission'] = df['Transmission'].replace('?', 'AUTO')
    df['Transmission'] = df['Transmission'].replace('Manual', 'MANUAL')
    
    ## WHEELTYPEID ##
    # replace by majority since data is categorical
    df['WheelTypeID'] = df['WheelTypeID'].replace('?', '1')
    df['WheelTypeID'] = pd.Categorical(df['WheelTypeID'])
    
    ## WHEEL TYPE ##
    # replace nan and ? into Alloy
    df['WheelType'] = df['WheelType'].replace(np.nan, 'Alloy')
    df['WheelType'] = df['WheelType'].replace('?', 'Alloy')
    
    ## NATIONALITY ##
    # replace '?' and 'USA' with 'AMERICAN'
    df['Nationality'] = df['Nationality'].replace('?', 'AMERICAN')
    df['Nationality'] = df['Nationality'].replace('USA', 'AMERICAN')
    
    ## SIZE ##
    # replace '?' into Medium
    df['Size'] = df['Size'].replace('?', 'MEDIUM')
    
    ## TOPTHREEAMERICANNAME ##
    # replace '?' with 'GM'
    df['TopThreeAmericanName'] = df['TopThreeAmericanName'].replace('?', 'GM')
    
    ## MMRAcquisitionAuctionAveragePrice ##
    # replace '?' with '0'
    df['MMRAcquisitionAuctionAveragePrice'] = df['MMRAcquisitionAuctionAveragePrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRAcquisitionAuctionAveragePrice'] = pd.to_numeric(df['MMRAcquisitionAuctionAveragePrice'])
    # fill the missing value with the mean of the column
    df['MMRAcquisitionAuctionAveragePrice'] = df['MMRAcquisitionAuctionAveragePrice'].fillna((df['MMRAcquisitionAuctionAveragePrice'].mean()))
    
    ## MMRAcquisitionAuctionCleanPrice ##
    # replace '?' with '0'
    df['MMRAcquisitionAuctionCleanPrice'] = df['MMRAcquisitionAuctionCleanPrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRAcquisitionAuctionCleanPrice'] = pd.to_numeric(df['MMRAcquisitionAuctionCleanPrice'])
    # fill the missing value with the mean of the column
    df['MMRAcquisitionAuctionCleanPrice'] = df['MMRAcquisitionAuctionCleanPrice'].fillna((df['MMRAcquisitionAuctionCleanPrice'].mean()))
    
    ## MMRAcquisitionRetailAveragePrice ##
    # replace '?' with '0'
    df['MMRAcquisitionRetailAveragePrice'] = df['MMRAcquisitionRetailAveragePrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRAcquisitionRetailAveragePrice'] = pd.to_numeric(df['MMRAcquisitionRetailAveragePrice'])
    # fill the missing value with the mean of the column
    df['MMRAcquisitionRetailAveragePrice'] = df['MMRAcquisitionRetailAveragePrice'].fillna((df['MMRAcquisitionRetailAveragePrice'].mean()))
    
    ## MMRAcquisitonRetailCleanPrice ##
    # replace '?' with '0'
    df['MMRAcquisitonRetailCleanPrice'] = df['MMRAcquisitonRetailCleanPrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRAcquisitonRetailCleanPrice'] = pd.to_numeric(df['MMRAcquisitonRetailCleanPrice'])
    # fill the missing value with the mean of the column
    df['MMRAcquisitonRetailCleanPrice'] = df['MMRAcquisitonRetailCleanPrice'].fillna((df['MMRAcquisitonRetailCleanPrice'].mean()))
    
    ## MMRCurrentAuctionAveragePrice ##
    # replace '?' with '0'
    df['MMRCurrentAuctionAveragePrice'] = df['MMRCurrentAuctionAveragePrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRCurrentAuctionAveragePrice'] = pd.to_numeric(df['MMRCurrentAuctionAveragePrice'])
    # fill the missing value with the mean of the column
    df['MMRCurrentAuctionAveragePrice'] = df['MMRCurrentAuctionAveragePrice'].fillna((df['MMRCurrentAuctionAveragePrice'].mean()))
    
    ## MMRCurrentAuctionCleanPrice ##
    # replace '?' with '0'
    df['MMRCurrentAuctionCleanPrice'] = df['MMRCurrentAuctionCleanPrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRCurrentAuctionCleanPrice'] = pd.to_numeric(df['MMRCurrentAuctionCleanPrice'])
    # fill the missing value with the mean of the column
    df['MMRCurrentAuctionCleanPrice'] = df['MMRCurrentAuctionCleanPrice'].fillna((df['MMRCurrentAuctionCleanPrice'].mean()))
    
    ## MMRCurrentRetailAveragePrice ##
    # replace '?' with '0'
    df['MMRCurrentRetailAveragePrice'] = df['MMRCurrentRetailAveragePrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRCurrentRetailAveragePrice'] = pd.to_numeric(df['MMRCurrentRetailAveragePrice'])
    # fill the missing value with the mean of the column
    df['MMRCurrentRetailAveragePrice'] = df['MMRCurrentRetailAveragePrice'].fillna((df['MMRCurrentRetailAveragePrice'].mean()))

    ## MMRCurrentRetailCleanPrice ##
    # replace '?' with '0'
    df['MMRCurrentRetailCleanPrice'] = df['MMRCurrentRetailCleanPrice'].replace('?', '0')
    # convert data type from string to numeric
    df['MMRCurrentRetailCleanPrice'] = pd.to_numeric(df['MMRCurrentRetailCleanPrice'])
    # fill the missing value with the mean of the column
    df['MMRCurrentRetailCleanPrice'] = df['MMRCurrentRetailCleanPrice'].fillna((df['MMRCurrentRetailCleanPrice'].mean()))
    
    ## MMRCurrentRetailRatio ##
    # replace '?' with '0'
    df['MMRCurrentRetailRatio'] = df['MMRCurrentRetailRatio'].replace('#VALUE!', '0')
    # convert data type from string to numeric
    df['MMRCurrentRetailRatio'] = pd.to_numeric(df['MMRCurrentRetailRatio'])
    # fill the missing value with the mean of the column
    df['MMRCurrentRetailRatio'] = df['MMRCurrentRetailRatio'].fillna((df['MMRCurrentRetailRatio'].mean()))

    ## VehBCost ##
    # replace '?' with '0'
    df['VehBCost'] = df['VehBCost'].replace('?', '0')
    # convert data type from string to numeric
    df['VehBCost'] = pd.to_numeric(df['VehBCost'])
    
    ## IsOnlineSale ##
    # replace '?' with '0'
    df['IsOnlineSale'] = df['IsOnlineSale'].replace('?', 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace(-1.0, 1)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace( 2, 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace( 4, 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace( 0.0, 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace( '0', 0)
    df['IsOnlineSale'] = df['IsOnlineSale'].replace( '1', 1)

    ## IsOnlineSale ##
    # convert data type from string to numeric
    df['IsOnlineSale'] = pd.Categorical(df['IsOnlineSale'])
    
    ## ForSale ##
    # replace noisy
    df['ForSale'] = df['ForSale'].replace('yes', 'Yes')
    df['ForSale'] = df['ForSale'].replace('YES', 'Yes')
    df['ForSale'] = df['ForSale'].replace('?', 'Yes')
    df['ForSale'] = df['ForSale'].replace('0', 'No')

    return df