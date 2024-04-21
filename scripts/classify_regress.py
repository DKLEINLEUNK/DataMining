import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

predictors = ['circumplex.valence', 'circumplex.arousal', 'activity', 'screen', 'appCat.communication', 'appCat.entertainment', 'appCat.social']


## Classification
# Algorithm 1: Gradient Descent
def conduct_XGB(data, predictors):
    # Define featuires (X) with the newly added variables and target (y)
    X = data[predictors]
    Y = data['mood_category']
    Y = np.array([1 if mood == 'above average' else 0 for mood in Y])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize features and fit SVM model
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = xgb.XGBClassifier()
    model.fit(X_train_scaled, y_train)

    # Predict on the testing set with the enhanced model
    y_pred = model.predict(X_test_scaled)
    # Evaluate the enhanced model
    accuracy = model.score(X_test_scaled, y_test)
    return (y_test, y_pred, accuracy)

# Algorithm 2: Support Vector Classification
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
    y_pred = model.predict(X_test_scaled)
    accuracy = model.score(X_test_scaled, y_test)
    return (y_test, y_pred, accuracy)

## Algorithm 3: RNN
# Helper function for RNN
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].value.values
        y = data.iloc[i + seq_length].value
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)


def conduct_RNN(data):
    # Define sequence length
    sequence_length = 5

    mood_data = data[data['variable'] == 'mood']
    mood_data_sorted = mood_data.sort_values(by=['id', 'time'])
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

    
    # Split the data into training and testing sets
    all_X, all_y = [], []
    for subject_data in sequences.values():
        all_X.extend(subject_data[0])
        all_y.extend(subject_data[1])

    all_X = np.array(all_X)
    all_y = np.array(all_y)
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    # Check the shapes of the resulting datasets
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    # Reshape input for LSTM layer
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    history = model.fit(X_train_reshaped, y_train, epochs=20, validation_data=(X_test_reshaped, y_test))

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)  # Use (y_pred_probs > 0.5).astype(int) for binary classification
    # accuracy = accuracy_score(y_test, y_pred)  # Adjust for binary classification
    # cm = confusion_matrix(y_test, y_pred)
    return y_pred, y_test, history

## Regression
# Check important features: Stepwise Regression
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



# Algorithm 1: Support Vector Regression
def conduct_SVR(data, predictors):
    X = data[predictors]
    Y = data['daily_mood']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    '''
    model = SVR()
    #Hyper parameter tuning using GridSearchCV
    # Parameter grid to test different hyperparameters
    param_grid = {
        'C': [0.1, 0.2, 0.5, 1, 2, 5, 1],  
        'gamma': ['scale', 'auto', 0.01, 0.1, 0.2],  
        'kernel': ['rbf', 'linear'],  
        'epsilon': [0.01, 0.1, 0.5, 1, 2, 5]  
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1)

    # Progress monitoring
    # with parallel_backend('threading', n_jobs=-1):  # Set n_jobs=-1 to use all available cores
    #     tqdm(grid_search.fit(X_train_scaled, y_train))
    grid_search.fit(X_train_scaled, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score (negative MSE):", grid_search.best_score_)
        # Best parameters and best score
    # Fit the model with the best hyperparameters
    '''
    model = SVR(C= 5, epsilon= 0.5, gamma= 0.01)

    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # print(y_pred)   
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("SVR - Mean Absolute Error: {:.2f}, R^2 Score: {:.2f}".format(mae, r2))
    return model, y_pred, y_test

    ## Algorithm 3: Random Forest Regression
def conduct_RFR(data, predictors):
    X = data[predictors]
    Y = data['daily_mood']

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    '''
    model = RandomForestRegressor()
    #Hyper parameter tuning using GridSearchCV
    # Parameter grid to test different hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],  
        'max_depth': [None, 10, 20, 30], 
        'min_samples_split': [2, 5, 10], 
        'min_samples_leaf': [1, 2, 4]
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=5, n_jobs=-1)

    # Progress monitoring
    # with parallel_backend('threading', n_jobs=-1):  # Set n_jobs=-1 to use all available cores
    #     tqdm(grid_search.fit(X_train_scaled, y_train))
    grid_search.fit(X_train_scaled, y_train)
    print("Best parameters:", grid_search.best_params_)
    print("Best score (negative MSE):", grid_search.best_score_)
    '''
    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Evaluation
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)

    print("Random Forest - MSE: {:.2f}, MAE: {:.2f}".format(rf_mse, rf_mae))

    return rf_model, rf_predictions, y_test


## Plotting
def create_confusion_matrix(y_test, y_pred, save_fig=False, name='confusion_matrix.png', binary=False):
    # Create confusion matrix
    if binary:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=[0, 1], yticklabels=[0, 1])
    else:
        cm = confusion_matrix(y_test, y_pred, labels=['below average', 'above average']) 
        sns.heatmap(cm, annot=True, fmt='g', xticklabels=['below average', 'above average'], yticklabels=['below average', 'above average'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_fig:
        plt.savefig(name)
    else:
        plt.show()

if __name__ == "__main__":
    # print(data.head())
    """ 
    mae, r2 = conduct_SVR(data, predictors)
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    print("SVR model has been trained and evaluated.") 
    accuracy = conduct_SVC(data, predictors)
    print(f"Accuracy: {accuracy}")
    """
    # y_pred, y_test, model = conduct_SVR()
    # y_test, y_pred, accuracy = conduct_DT(data_clean, predictors)
    
    # print(f"Accuracy: {accuracy}")
    data_clean = fix_features.get_clean_data()
    data = fix_features.read_data()

    model, y_pred, y_test = conduct_SVR(data_clean, predictors)
    # create_confusion_matrix(y_test, y_pred)
    # history = conduct_RNN(data, predictors)
