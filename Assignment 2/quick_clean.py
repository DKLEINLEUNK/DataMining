import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
os.chdir("C:\\Users\\Kaleem\\Documents\\Courses\\Data Mining\\DataMining\\Assignment 2")

# Read the CSV file
df = pd.read_csv('data\\train.csv')
# df.to_csv('data\\train_1000.csv', index=False)
print("Data read successfully")

#####################################################################################
#####      Data Preprocessing
df['date_time'] = pd.to_datetime(df['date_time'])
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
###### Convert columns to categorical
categorical_cols = [
    'srch_id', 'promotion_flag', 'random_bool', 'prop_id',
    'visitor_location_country_id', 'srch_destination_id', 'prop_country_id'
]
for col in categorical_cols:
    df[col] = df[col].astype('category')

###### Convert columns to boolean
boolean_cols= ['promotion_flag', 'random_bool', 'booking_bool']

for col in boolean_cols:
    df[col] = df[col].astype(bool)
# Drop original 'date_time' column
df.drop(columns=['date_time'], inplace=True)

# Drop columns with more than 70% missing values
threshold = 0.7
df = df.loc[:, df.isnull().mean() < threshold]

# Remove columns with no variance
nunique = df.apply(pd.Series.nunique)
df = df.loc[:, nunique != 1]
print("Data Preprocessing done successfully")
#########################################################################################
#####     Imputation
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object, 'category']).columns

numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Remaining columns 
remaining_columns = df.columns.tolist()
print("Remaining columns:", remaining_columns)
###########################################################################################
#####     Data Splitting (and oversampling from booked pages)
X = df.drop(columns=['booking_bool'])
y = df['booking_bool']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled data
df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='booking_bool')], axis=1)

# Display the cleaned dataset information
df_resampled.info(), df_resampled.describe()

####  Save the cleaned dataset
df_resampled.to_csv('data\\train_cleaned.csv', index=False)
print("train_cleaned.csv created successfully in data folder")



###########################################################################
###### TEST SET
###########################################################################
# Read the CSV file
df = pd.read_csv('data\\test.csv')
# df.to_csv('data\\train_1000.csv', index=False)
print("Data read successfully")

#####################################################################################
#####      Data Preprocessing
df['date_time'] = pd.to_datetime(df['date_time'])
df['year'] = df['date_time'].dt.year
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.day
df['hour'] = df['date_time'].dt.hour
###### Convert columns to categorical
categorical_cols = [
    'srch_id', 'promotion_flag', 'random_bool', 'prop_id',
    'visitor_location_country_id', 'srch_destination_id', 'prop_country_id'
]
for col in categorical_cols:
    df[col] = df[col].astype('category')

###### Convert columns to boolean
boolean_cols= ['promotion_flag', 'random_bool']

for col in boolean_cols:
    df[col] = df[col].astype(bool)
# Drop original 'date_time' column
df.drop(columns=['date_time'], inplace=True)

# Drop columns with more than 70% missing values
threshold = 0.7
df = df.loc[:, df.isnull().mean() < threshold]

# Remove columns with no variance
nunique = df.apply(pd.Series.nunique)
df = df.loc[:, nunique != 1]
print("Data Preprocessing done successfully")
#########################################################################################
#####     Imputation
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object, 'category']).columns

numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Remaining columns 
remaining_columns = df.columns.tolist()
print("Remaining columns:", remaining_columns)
###########################################################################################
#####     Data Splitting (and oversampling from booked pages)
df.to_csv('data\\test_cleaned.csv', index=False)
print("test_cleaned.csv created successfully in data folder")
