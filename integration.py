import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import numpy as np
import xgboost as xgb

class TestIntegration(unittest.TestCase):
    def test_integration(self):
        data = pd.read_csv("sepsis dataset.csv")
        data = data.drop(columns=['Gender'])
        data[['Systolic', 'Diastolic']] = data['BloodPressure'].str.split('/', expand=True)
        data['Systolic'] = data['Systolic'].astype(float)
        data['Diastolic'] = data['Diastolic'].astype(float)
        data = data.drop(columns=['BloodPressure'])
        
        def create_lstm_model(input_shape):
            model = Sequential([
                LSTM(64, input_shape=input_shape),
                Dense(1, activation='sigmoid')
            ])
            return model
        
        X = data.drop(columns=['Sepsis']).astype(float)
        y = data['Sepsis'].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        lstm_model = create_lstm_model(input_shape=(X_train.shape[1], 1))
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=1, batch_size=1, verbose=0)

        def create_rnn_model(input_shape):
            model = Sequential([
                Dense(64, activation='relu', input_shape=input_shape),
                Dense(1, activation='sigmoid')
            ])
            return model

        rnn_model = create_rnn_model(input_shape=(X_train.shape[1],))
        rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        rnn_model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)

        lstm_predictions = lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
        rnn_predictions = rnn_model.predict(X_test)
        
        combined_predictions = np.concatenate((lstm_predictions, rnn_predictions), axis=1)
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(combined_predictions, y_test)
        integration_successful = True

        self.assertTrue(integration_successful)
        print("Test case passed.")

if __name__ == '__main__':
    unittest.main()
