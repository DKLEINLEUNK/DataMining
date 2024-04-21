import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add path to the function
def read_data(path=None):
    '''
    Reads the dataset and returns it.
    Uses absolute path to read the data.
    
    returns:
    - data: pandas DataFrame
    '''
    if path:
        data = pd.read_csv(path)
    else:
        data = pd.read_csv('/courses/Data Mining/DataMining/data/mood_smartphone.csv')  
    data['time'] = pd.to_datetime(data['time'])
    data['date'] = data['time'].dt.date
    return data


# Define a function to calculate rolling features individually
def calculate_rolling_features(grouped_data, feature_name, window_size=6):  # Last five days of mood excluding current day
    return grouped_data.rolling(window=window_size, min_periods=6).agg({
        'mean': lambda x: x.shift(1).mean()
        #'min': lambda x: x.shift(1).min(),
        #'max': lambda x: x.shift(1).max(),
        #'std': lambda x: x.shift(1).std()
    }).rename(columns={
        'mean': f'avg_{feature_name}'
        #'min': f'min_{feature_name}_last_5_days',
        #'max': f'max_{feature_name}_last_5_days',
        #'std': f'std_{feature_name}_last_5_days'
    })


def feature_engineer(data):
    '''
    Feature engineers the data to include daily mood and calculate rolling features for the 'mood' variable for 
    the last five days. 
    '''     
    daily_mood = data[data['variable'] == 'mood'].groupby(['id', 'date']).agg(daily_mood=('value', 'mean')).reset_index()
    # Calculate rolling features for 'mood' variable
    mood_features = calculate_rolling_features(daily_mood.groupby('id')['daily_mood'], 'mood')

    # Merge rolling features with the original data
    daily_mood_features = pd.concat([daily_mood.reset_index(drop=True), mood_features.reset_index(drop=True)], axis=1)
    daily_mood_features.head(10) 
    return daily_mood_features

list_features = ['circumplex.arousal', 'circumplex.valence', 'activity', 'screen', 'appCat.communication', 'appCat.entertainment', 'appCat.social']

def add_features(daily_mood_features, list_features=list_features):
    '''
    Adds features to the data by calculating the daily average of selected features.
    '''
    data = read_data()
    daily_aggregates = data[data['variable'].isin(list_features)].groupby(['id', 'date', 'variable']).agg(daily_average=('value', 'mean')).reset_index()
    # Pivot the aggregated data so each variable becomes a column
    daily_pivoted = daily_aggregates.pivot_table(index=['id', 'date'], columns='variable', values='daily_average').reset_index()
    # Merge the pivoted data with the daily mood data to get a single dataframe with all features and the target
    full_data = daily_mood_features.merge(daily_pivoted, on=['id', 'date'], how='left')
    # Display the merged dataset structure
    return full_data

def impute_missing_values(full_data):
    '''
    Imputes missing values in the data.
    '''
    data = read_data()
    missing_values = full_data.isnull().sum()
    # Drop columns with more than 30% missing values
    columns_to_drop = missing_values[missing_values > len(data) * 0.3].index
    data_cleaned = full_data.drop(columns=columns_to_drop)

    # Impute missing values with the median for the remaining columns
    for column in data_cleaned.columns:
        if data_cleaned[column].isnull().any():
            median_value = data_cleaned[column].median()
            data_cleaned[column].fillna(median_value, inplace=True)
    
    # Re-check for missing values to ensure all are handled
    data_cleaned.isnull().sum()
    return data_cleaned

# Scaling all numerical variables
def scale_variables(data):
    # scale all variables in dataset    
    numeric_cols = data.select_dtypes(include=['number']).columns
    # Initialize the scaler
    scaler = StandardScaler()
    # Fit and transform the data
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

     # Function to classify the value
    def classify_value(value):
        if value < 0:
            return "below average"
        else:
            return "above average"

    # Apply the function to create a new column based on 'NumericValue'
    data['mood_category'] = data['daily_mood'].apply(classify_value)
    return data

def get_clean_data(path=None):
    data_cleaned = scale_variables(impute_missing_values(add_features(feature_engineer(read_data(path=path)))))
    return data_cleaned


if __name__ == "__main__":
    data = read_data()
    daily_mood_features = feature_engineer(data)
    print("Data has been now includes last 5 days of mood.")
    print(daily_mood_features.head(10))
    full_data = add_features(daily_mood_features)
    print("Features have been added.")
    print("Missing values have been imputed.")
    print(len(full_data))
    data_cleaned = impute_missing_values(full_data)
    scaled_data = scale_variables(data_cleaned)
    scaled_data[['id', 'date', 'daily_mood', 'mood_category', 'avg_mood', 'circumplex.valence', 'circumplex.arousal']].tail(5).to_latex('data.tex')
    print(scaled_data.head(10))

