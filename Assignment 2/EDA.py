""" Helper functions for the Exploratory Data Analysis notebook """

import pandas as pd


def load_data_in_chunks(path, chunk_size=1000):
    """Load data in chunks"""
    chunks = pd.read_csv(path, chunksize=chunk_size)
    return chunks


def get_head(path, n=5):
    """Get the first n rows of a dataset"""
    chunks = load_data_in_chunks(path)
    for chunk in chunks:
        return chunk.head(n)


def apply_function_to_chunks(path, function):
    """Apply a function to each chunk"""
    chunks = load_data_in_chunks(path)
    return [function(chunk) for chunk in chunks]


def count_missing_values(df):
    """Count missing values in a DataFrame (chunck)"""
    return df.isnull().sum()


def count_missing_values_by_column(df):
    """Count missing values by column in a DataFrame (chunk)"""
    return df.isnull().sum(axis=0)


def get_unique_values(path, column):
    """Get unique values in a DataFrame (chunk)"""
    chunks = load_data_in_chunks(path)
    unique_values = set()
    for chunk in chunks:
        unique_values.update(chunk[column].unique())
    return unique_values


def get_value_counts(path, value_col, counts_col):
    """Get value counts in a DataFrame (chunk)"""
    chunks = load_data_in_chunks(path)
    unique = set()
    for chunk in chunks:
        chunk[value_col]