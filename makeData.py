# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 01:53:54 2019

@author: deepak
"""

import pandas as pd
import numpy as np

def makeFeatures(dataframe) :
    train_df = dataframe
    page_details = pd.DataFrame([i.split("_")[-3:] for i in train_df["Page"]])
    page_details.columns = ["project", "access", "agent"]
    
    
    #Getting Unique values for each column of page_details dataframe
    project_columns = page_details['project'].unique()
    access_columns = page_details['access'].unique()
    agents_columns = page_details['agent'].unique()
    #Creating a list for each column
    A = list(range(0, len(project_columns)))
    B = list(range(0, len(access_columns)))
    C = list(range(0, len(agents_columns)))
    #Creating a dictionary for each column with integer as value
    project_columns_dict = dict(zip(project_columns,A))
    access_columns_dict = dict(zip(access_columns,B))
    agents_columns_dict = dict(zip(agents_columns,C))
    #Mapping values encoded values to page_details dataframe
    page_details['project'] = page_details['project'].map(project_columns_dict)
    page_details['access'] = page_details['access'].map(access_columns_dict)
    page_details['agent'] = page_details['agent'].map(agents_columns_dict)
    

    #Preprocessing train_df by removing unwanted data
    #Dropping Page Column
    train_df.drop('Page', axis=1, inplace=True)
    #Dropping rows where all the values  are NA
    train_df = train_df.dropna(how='all')
    #Filling NA values with 0 for other pages
    train_df = train_df.fillna(0)
    
    #setting date, week, weekend, weekday, month, quarter, half year, year from date
    #This will help in setting the time series features for our page
    Date = train_df.columns.values.tolist()
    date_df = pd.DataFrame(Date,columns=['date'])
    date_df['date'] = date_df['date'].astype('datetime64[ns]')
    date_df['weekend'] = ((date_df.date.dt.dayofweek) // 5 == 1).astype(float)
    date_df['weekday'] = date_df['date'].apply(lambda x: x.weekday())
    date_df['year']=date_df.date.dt.year
    date_df['month']=date_df.date.dt.month 
    date_df['day']=date_df.date.dt.day
    date_df['quarter'] = date_df.date.dt.quarter
    date_df['halfYear'] = date_df['quarter'].apply(lambda x: 1 if x < 3 else 2)
    
    #Creating another dataframe of date_df to use it for manipulation
    date_df_feature = date_df.set_index(date_df.columns[0])
    date_df_feature.index.name = None
    
    #Getting Unique values for column - year dataframe - data
    year_columns = date_df_feature['year'].unique()
    #Creating a list for  column
    D = list(range(0, len(year_columns)))
    #Creating a dictionary for each column with integer as value
    year_columns_dict = dict(zip(year_columns,D))
    #Mapping values encoded values to page_details dataframe
    date_df_feature['year'] = date_df_feature['year'].map(year_columns_dict)
    
    return train_df, date_df_feature, page_details

def setSamples(start, end, train_dfp, date_df_featurep, page_detailsp) :
    train_df = train_dfp
    date_df_feature = date_df_featurep
    page_details = page_detailsp
    alter_train_df = train_df.T
    
    dataset_o = np.empty((0,train_df.shape[1],25), np.float64)
    
    #Lets calculate median and mean for page till the given date range
    # Median by page
    pagemedian = train_df.median(axis=1)

    # Average by page
    pagemean = train_df.mean(axis=1)
   
    #Lets Calculate Min and Max for page till the given date range
    # Max for page
    pagemax = train_df.max(axis=1)

    # Min for page
    pagemin = train_df[train_df > 0].min(axis=1) 
    
    #set number of rows in data to proceed with training
    count  = end - start + 1
    for setRowNum in range(count) :
        data = date_df_feature.merge(pd.DataFrame(alter_train_df.loc[:,setRowNum]), how="inner", left_index=True, right_index=True)
        data.columns.values[-1] = "visits"
        
        #Setting median values for all date features taken in date_df_feature

        #Median for page
        data['median'] = pagemedian.loc[setRowNum]

        #Median for a weekday
        w_median = data.groupby(['weekday'])['visits'].median()
        data['wMedian'] = data['weekday'].map(w_median.to_dict())

        #Median for a Month
        data['yearMonth'] = data['year'].astype(str) + ',' + data['month'].astype(str)
        m_median = data.groupby(['yearMonth'])['visits'].median()
        data['mMedian'] = data['yearMonth'].map(m_median.to_dict())
        data.drop('yearMonth', axis=1, inplace=True)

        #Median for a Quarter
        q_median = data.groupby(['quarter'])['visits'].median()
        data['qMedian'] = data['quarter'].map(q_median.to_dict())

        #Median for a HalfYear
        hy_median = data.groupby(['halfYear'])['visits'].median()
        data['hyMedian'] = data['halfYear'].map(hy_median.to_dict())

        #Median for a Year
        y_median = data.groupby(['year'])['visits'].median()
        data['yMedian'] = data['year'].map(y_median.to_dict())
        
        #Setting mean values for all date features taken in date_df_feature

        #Mean for page
        data['mean'] = pagemean.loc[setRowNum]

        #Mean for a Weekday
        w_mean = data.groupby(['weekday'])['visits'].mean()
        data['wMean'] = data['weekday'].map(w_mean.to_dict())

        #Mean for a Month
        data['yearMonth'] = data['year'].astype(str) + ',' + data['month'].astype(str)
        m_mean = data.groupby(['yearMonth'])['visits'].mean()
        data['mMean'] = data['yearMonth'].map(m_mean.to_dict())
        data.drop('yearMonth', axis=1, inplace=True)

        #Mean for a Quarter
        q_mean = data.groupby(['quarter'])['visits'].mean()
        data['qMean'] = data['quarter'].map(q_mean.to_dict())

        #Mean for a HalfYear
        hy_mean = data.groupby(['halfYear'])['visits'].mean()
        data['hyMean'] = data['halfYear'].map(hy_mean.to_dict())

        #Mean for a Year
        y_mean = data.groupby(['year'])['visits'].mean()
        data['yMean'] = data['year'].map(y_mean.to_dict())
        
        #setting Min and Max Values in data

        # Min Column in Data
        data['min'] =  pagemin.loc[setRowNum]

        # Max Column in Data
        data['max'] = pagemax.loc[setRowNum]
        
        data['lang'] = page_details['project'].loc[setRowNum]
        data['access'] = page_details['access'].loc[setRowNum]
        data['agent'] = page_details['agent'].loc[setRowNum]
        
        data = data[['visits', 'median', 'wMedian', 'mMedian', 'qMedian', 'hyMedian', 
                     'yMedian', 'mean', 'wMean', 'mMean', 'qMean', 'hyMean', 'yMean', 
                     'min', 'max', 'lang', 'access', 'agent', 'year', 'halfYear', 
                     'quarter', 'month', 'day', 'weekday', 'weekend']]
        
        # Set dataframe in array 
        arrayData = data.values
        arrayData = np.reshape(arrayData, (1, arrayData.shape[0], arrayData.shape[1]))
        dataset_o = np.append(dataset_o, arrayData, axis=0)
        dataset = np.log1p(dataset_o)
    return dataset_o, dataset
        
        
        
        
        
    