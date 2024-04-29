import unittest
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.model_selection import train_test_split
import xgboost as xgb

class TestPrediction(unittest.TestCase):
    def setUp(self):
        # Step 1: Load the dataset from CSV file
        self.data = pd.read_csv("sepsis dataset.csv")

        # Remove the 'Gender' column
        self.data = self.data.drop(columns=['Gender'])

        # Step 2: Preprocess the dataset
        # Split the 'BloodPressure' column into 'Systolic' and 'Diastolic' columns
        self.data[['Systolic', 'Diastolic']] = self.data['BloodPressure'].str.split('/', expand=True)

        # Convert 'Systolic' and 'Diastolic' columns to float
        self.data['Systolic'] = self.data['Systolic'].astype(float)
        self.data['Diastolic'] = self.data['Diastolic'].astype(float)

        # Drop the original 'BloodPressure' column
        self.data = self.data.drop(columns=['BloodPressure'])

        # Step 4: Prepare the data for training
        X = self.data.drop(columns=['Sepsis']).astype(float)
        y = self.data['Sepsis'].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 5: Create and train the LSTM model
        self.lstm_model = self.create_lstm_model(input_shape=(X_train.shape[1], 1))
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=32, verbose=0)

        # Step 7: Train the RNN model
        self.rnn_model = self.create_rnn_model(input_shape=(X_train.shape[1],))
        self.rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.rnn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Step 10: Train XGBoost model
        lstm_predictions = self.lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
        rnn_predictions = self.rnn_model.predict(X_test)
        combined_predictions = np.concatenate((lstm_predictions, rnn_predictions), axis=1)
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.fit(combined_predictions, y_test)

    def create_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape),
            Dense(1, activation='sigmoid')
        ])
        return model

    def create_rnn_model(self, input_shape):
        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(1, activation='sigmoid')
        ])
        return model

    def test_prediction_no_sepsis(self):
        # Simulate input data for a patient with no sepsis
        user_data = pd.DataFrame({
            'Age': [35],
            'HeartRate': [75],
            'Temperature': [36.8],
            'Systolic': [120],
            'Diastolic': [80],
            'RespiratoryRate': [16],
            'WhiteBloodCellCount': [7000],
            'LacticAcid': [1.2]
        })

        lstm_prediction = self.lstm_model.predict(user_data.values.reshape(1, user_data.shape[1], 1))
        rnn_prediction = self.rnn_model.predict(user_data)
        combined_prediction = np.concatenate((lstm_prediction, rnn_prediction), axis=1)
        final_prediction = self.xgb_model.predict(combined_prediction)[0]

        # Expecting no sepsis detected
        self.assertEqual(final_prediction, 0, "Expected no sepsis detected")
        print("Test case 'No Sepsis' passed.")
        print("Sepsis prediction: No sepsis detected.")

    def test_prediction_sepsis_detected(self):
        # Simulate input data for a patient with sepsis
        user_data = pd.DataFrame({
            'Age': [55],
            'HeartRate': [90],
            'Temperature': [38.5],
            'Systolic': [110],
            'Diastolic': [70],
            'RespiratoryRate': [20],
            'WhiteBloodCellCount': [15000],
            'LacticAcid': [3.5]
        })

        lstm_prediction = self.lstm_model.predict(user_data.values.reshape(1, user_data.shape[1], 1))
        rnn_prediction = self.rnn_model.predict(user_data)
        combined_prediction = np.concatenate((lstm_prediction, rnn_prediction), axis=1)
        final_prediction = self.xgb_model.predict(combined_prediction)[0]

        # Expecting sepsis detected
        self.assertEqual(final_prediction, 1, "Expected sepsis detected")
        print("Test case 'Sepsis Detected' passed.")
        print("Sepsis prediction: Sepsis detected.")

if __name__ == '__main__':
    unittest.main()
