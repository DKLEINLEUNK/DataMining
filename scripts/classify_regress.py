import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
from .fix_features import *
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import statsmodels.api as sm

predictors = ['circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'appCat.communication', 'appCat.entertainment', 'appCat.social']
data = get_clean_data()

## Classification
# Algorithm 1: Support Vector Regression
def conduct_SVC(data, predictors):
    # Define featuires (X) with the newly added variables and target (y)
    X = data[predictors]
    Y = data['mood_category']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize features and fit SVM model
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = SVC()
    model.fit(X_train_scaled, y_train)

    # Predict on the testing set with the enhanced model
    y_pred = model.predict(X_test_scaled)
    # Evaluate the enhanced model
    accuracy = model.score(X_test_scaled, y_test)
    return (y_test, y_pred, accuracy)

# Algorithm 2: RNN
def conduct_RNN(data, predictors):
    # Define sequence length
    sequence_length = 5

    # Function to create sequences
    def create_sequences(data, seq_length):
        xs = []
        ys = []

        for i in range(len(data) - seq_length):
            x = data.iloc[i:(i + seq_length)].value.values
            y = data.iloc[i + seq_length].value
            xs.append(x)
            ys.append(y)
        
        return np.array(xs), np.array(ys)

    mood_data = data[data['variable'] == 'mood']
    mood_data_sorted = mood_data.sort_values(by=['id', 'time'])
    
    # Normalize mood values for each subject
    scaler = StandardScaler()
    mood_data_sorted['value'] = mood_data_sorted.groupby('id')['value'].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

    # Create sequences
    sequences = {}
    for subject in mood_data_sorted['id'].unique():
        subject_data = mood_data_sorted[mood_data_sorted['id'] == subject]
        X, y = create_sequences(subject_data, sequence_length)
        sequences[subject] = (X, y)

    # Example of sequences for the first subject
    first_subject_id = list(sequences.keys())[0]
    sequences[first_subject_id][0][:5], sequences[first_subject_id][1][:5]
    model = Sequential([
    LSTM(50, input_shape=(sequence_length, 1)),  # LSTM layer with 50 units
    Dense(1)  # Output layer to predict the next mood value
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Split the data into training and testing sets
    X = data[predictors]
    Y = data['mood_category']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Reshape input for LSTM layer
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    history = model.fit(X_train_reshaped, y_train, epochs=20, validation_data=(X_test_reshaped, y_test))

    return history

## Regression

# Algorithm 1: Support Vector Regression
def conduct_SVR(data, predictors):
    # Define featuires (X) with the newly added variables and target (y)
    X = data[predictors]
    Y = data['daily_mood']

    # Split the enhanced data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize enhanced features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Re-train an SVM model with the enhanced feature set
    model = SVR()
    model.fit(X_train_scaled, y_train)

    # Predict on the testing set with the enhanced model
    y_pred = model.predict(X_test_scaled)

    # Evaluate the enhanced model
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, r2

def forward_selection(data, response):
    excluded_columns = ['id', 'date']
    data = data.drop(columns=excluded_columns)
    data.columns = [col.replace('.', '_') for col in data.columns]
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response,
                                       ' + '.join(selected + [candidate]))
            score = sm.OLS.from_formula(formula, data).fit().aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {}".format(response, ' + '.join(selected))
    model = sm.OLS.from_formula(formula, data).fit()
    return model


def create_confusion_matrix(y_test, y_pred, save_fig=False):
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['bad', 'good'])
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['bad', 'good'], yticklabels=['bad', 'good'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_fig:
        plt.savefig('confusion_matrix.png')
    else:
        plt.show()

if __name__ == "__main__":
    print(data.head())
    """ 
    mae, r2 = conduct_SVR(data, predictors)
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    print("SVR model has been trained and evaluated.") 
    accuracy = conduct_SVC(data, predictors)
    print(f"Accuracy: {accuracy}")
    """
    # history = conduct_RNN(data, predictors)
    