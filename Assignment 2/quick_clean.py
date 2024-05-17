import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from xgboost_train import timing_decorator
import dask.dataframe as dd
from sklearn.datasets import make_classification
from collections import Counter


@timing_decorator
def preprocessing(df, scale=False, train=False, smote_flag=False):
    #####      Data Preprocessing
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['year'] = df['date_time'].dt.year
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df.drop(columns=['date_time'], inplace=True)
    ###### Convert columns to categorical
    categorical_cols = [
        'srch_id', 'promotion_flag', 'random_bool', 'prop_id', 'site_id', 
        'visitor_location_country_id', 'srch_destination_id', 'prop_country_id'
    ]
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    ###### Convert columns to boolean
    boolean_cols= ['promotion_flag', 'random_bool', 'srch_saturday_night_bool', 'prop_brand_bool']

    for col in boolean_cols:
        df[col] = df[col].astype(bool)

    # Drop columns with more than 70% missing values
    threshold = 0.7
    df = df.loc[:, df.isnull().mean() < threshold]

    # Remove columns with no variance
    nunique = df.apply(pd.Series.nunique)
    df = df.loc[:, nunique != 1]

    # Scale numeric columns
    # Create a DataFrame
    # Select only the numeric columns
    if scale:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = numeric_cols.drop(['year', 'month', 'day', 'hour'])
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit and transform the numeric data
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


    print("Data Preprocessing done successfully")
    
    ## Imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object, 'category']).columns

    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    ## Remaining columns 
    remaining_columns = df.columns.tolist()
    print("Remaining columns:", remaining_columns)
    ## Data Splitting (and oversampling from booked pages)
    
    if train:
        df["booking_bool"] = df["booking_bool"].astype(bool)
        df["click_bool"] = df["click_bool"].astype(bool)
        
        if smote_flag:
            X = df.drop(columns=['booking_bool'])
            y = df['booking_bool']
            smote = SMOTE(sampling_strategy=0.5, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            print('Original dataset shape %s' % Counter(y))
            print('Resampled dataset shape %s' % Counter(y_resampled))
            
            # Combine the resampled data
            df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='booking_bool')], axis=1)
            df_resampled.head(10)

            # Display the cleaned dataset information
            df_resampled.info(), df_resampled.describe()
            ####  Save the cleaned dataset
        else: 
            df_resampled = df
            df_resampled.info(), df_resampled.describe()
    else: 
        df_resampled = df
        df_resampled.info(), df_resampled.describe()
    return df_resampled

if __name__ == "__main__":
    os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")
    
    df_train = pd.read_csv('data\\train.csv')
    df_test = pd.read_csv('data\\test.csv')
    
    
    train_cleaned = preprocessing(df_train, scale=False, train=True, smote_flag=True)
    train_cleaned.to_csv('data\\train_cleaned.csv', index=False)
    print("Train data saved successfully")
    
    test_cleaned = preprocessing(df_test, scale=False)
    test_cleaned.to_csv('data\\test_cleaned.csv', index=False)
    print("Test data saved successfully")