import numpy as np
import pandas as pd

df = pd.read_csv("data/mood_smartphone.csv")

# for each variable, impute missing values with the median of the variable
df['value'] = df.groupby('variable')['value'].transform(lambda x: x.fillna(x.median()))

# for each variable in column 'variable', standardize the values
df['value'] = df.groupby('variable')['value'].transform(lambda x: (x - x.mean()) / x.std())

# remove outliers using 3 standard deviations
df = df[np.abs(df['value'] - df['value'].mean()) <= (3 * df['value'].std())]

