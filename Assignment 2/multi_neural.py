import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
import logging
from tensorflow.keras import backend as K
from helper import timing_decorator

logging.basicConfig(filename='neural_network.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Load the new dataset with outcome variables

def get_data(PATH_train, PATH_test, test=False):
    data_new = pd.read_csv(PATH_train)
    # data_sample = data_new.sample(n=5000, random_state=42)
    features = data_new.drop(columns=['position', 'click_bool', 'booking_bool', 'srch_id'])
    test = pd.read_csv(PATH_test)
    df_predict = test.drop(columns=["srch_id"], inplace=False)
    predictors_train = features.columns
    predictors_test = df_predict.columns
    common_cols = list(set(predictors_train).intersection(predictors_test))

    # Separate the feature columns and target columns correctly
    features = features[common_cols]
    # target_position = data_new['position']
    target_click_bool = data_new['click_bool']
    target_booking_bool = data_new['booking_bool']

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if test:
        X_train = features_scaled
        X_test = pd.read_csv(PATH_test)
        test_scaled = scaler.fit_transform(test)


        # y_test_position = y_test.iloc[:, 0]
        y_test_click = X_test.iloc[:, 1]
        y_test_booking = y_test.iloc[:, 2]
        y_train = pd.concat([target_click_bool, target_booking_bool], axis=1)

        # y_train_position = y_train.iloc[:, 0]
        y_train_click = y_train.iloc[:, 0]
        y_train_booking = y_train.iloc[:, 1]
        print("Data loaded and feature engineered")
        return X_train, y_train_click, y_train_booking, y_test_click, y_test_booking

    else:
        X_train, X_test, y_train, y_test = train_test_split(features_scaled,
                                                            pd.concat([target_click_bool, target_booking_bool], axis=1),
                                                            test_size=0.2, random_state=42)
        # Prepare the data
        y_train = pd.concat([target_click_bool, target_booking_bool], axis=1)

        # y_train_position = y_train.iloc[:, 0]
        y_train_click = y_train.iloc[:, 0]
        y_train_booking = y_train.iloc[:, 1]

        print("Data loaded and feature engineered")
        return X_train, y_train_click, y_train_booking, X_test, y_test


@timing_decorator
def neural_network(X_train, y_train_click, y_train_booking, X_test, y_test):
    # Define the neural network architecture
    input_layer = Input(shape=(X_train.shape[1],))
    hidden_layer_1 = Dense(128, activation='relu')(input_layer)
    hidden_layer_2 = Dense(64, activation='relu')(hidden_layer_1)
    hidden_layer_3 = Dense(32, activation='relu')(hidden_layer_2)
    output_click = Dense(1, activation='sigmoid', name='click')(hidden_layer_3)
    output_booking = Dense(1, activation='sigmoid', name='booking')(hidden_layer_3)

    # Create the model
    model = Model(inputs=input_layer, outputs=[output_click, output_booking])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'click': BinaryCrossentropy(), 'booking': BinaryCrossentropy()},
                  metrics={'click': 'accuracy', 'booking': 'accuracy'})

    # Train the model
    model.fit(X_train, {'click': y_train_click, 'booking': y_train_booking}, epochs=10, batch_size=64)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    click_accuracy = accuracy_score(y_test.iloc[:, 0], y_pred[0])
    booking_accuracy = accuracy_score(y_test.iloc[:, 1], y_pred[1])
    confusion_matrix_click = confusion_matrix(y_test.iloc[:, 0], y_pred[0])
    confusion_matrix_booking = confusion_matrix(y_test.iloc[:, 1], y_pred[1])

    print("Click Accuracy:", click_accuracy)
    print("Booking Accuracy:", booking_accuracy)
    print("Confusion Matrix Click:\n", confusion_matrix_click)
    print("Confusion Matrix Booking:\n", confusion_matrix_booking)

    logging.info(f"Click Accuracy: {click_accuracy}")
    logging.info(f"Booking Accuracy: {booking_accuracy}")
    logging.info(f"Confusion Matrix Click:\n{confusion_matrix_click}")
    logging.info(f"Confusion Matrix Booking:\n{confusion_matrix_booking}")
    return y_pred, model


if __name__ == "__main__":
    PATH_train = '/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2/data/train_cleaned.csv'
    PATH_test = '/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2/data/test_cleaned.csv'
    PATH_csv = '/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2/pending_submissions/neural_network.csv'
    PATH_compare = '/windows/Users/Kaleem/Documents/Courses/Data Mining/DataMining/Assignment 2/pending_submissions/never_mind/neural_network.csv'

    X_train, y_train_click, y_train_booking, X_test, y_test = get_data(PATH_train=PATH_train, PATH_test=PATH_test)

    y_pred, model = neural_network(X_train, y_train_click, y_train_booking, X_test, y_test)

    # X_train, y_train_click, y_train_booking, test_scaled = get_data(test=True)
    # Define the neural network architecture
#
# submission_df_compare = pd.read_csv(PATH_compare)
# submission_df_check = submission_df_compare.sort_values(by="booking_bool", ascending=False).drop(columns=['click_bool', 'booking_bool'])
# submission_df_check.head(10)
# submission_df_check.to_csv(PATH_compare, index=False)
#
# len(submission_df_check)
#
# submission_df_compare.head(10)
