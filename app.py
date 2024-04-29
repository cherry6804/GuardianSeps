from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.model_selection import train_test_split
import xgboost as xgb

app = Flask(__name__)

# Load the dataset from CSV file
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

# Prepare the data for training
X = data.drop(columns=['Sepsis']).astype(float)
y = data['Sepsis'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LSTM model
lstm_model = create_lstm_model(input_shape=(X_train.shape[1], 1))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=32)

# Define the RNN model creation function
def create_rnn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    return model

# Train the RNN model
rnn_model = create_rnn_model(input_shape=(X_train.shape[1],))
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Extract predictions from LSTM and RNN models
lstm_predictions = lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
rnn_predictions = rnn_model.predict(X_test)
combined_predictions = np.concatenate((lstm_predictions, rnn_predictions), axis=1)

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(combined_predictions, y_test)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot password.html')

@app.route('/app')
def show_app():
    return render_template('app.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    heart_rate = int(request.form['heartRate'])
    temperature = float(request.form['temperature'])
    blood_pressure = request.form['bloodPressure']
    systolic, diastolic = map(float, blood_pressure.split('/'))
    respiratory_rate = int(request.form['respiratoryRate'])
    white_blood_cell_count = int(request.form['whiteBloodCellCount'])
    lactic_acid = float(request.form['lacticAcid'])

    user_data = pd.DataFrame({
        'Age': [age],
        'HeartRate': [heart_rate],
        'Temperature': [temperature],
        'Systolic': [systolic],
        'Diastolic': [diastolic],
        'RespiratoryRate': [respiratory_rate],
        'WhiteBloodCellCount': [white_blood_cell_count],
        'LacticAcid': [lactic_acid]
    })

    lstm_prediction = lstm_model.predict(user_data.values.reshape(1, user_data.shape[1], 1))
    rnn_prediction = rnn_model.predict(user_data)
    combined_prediction = np.concatenate((lstm_prediction, rnn_prediction), axis=1)
    final_prediction = xgb_model.predict(combined_prediction)[0]

    if final_prediction == 0:
        print("Prediction: No sepsis detected.")
        return "No sepsis detected."
    elif final_prediction == 1:
        print("Prediction: Sepsis detected.")
        return "Sepsis detected."

    # If none of the conditions are met, return an error message
    return "Error: Prediction failed."

if __name__ == '__main__':
    app.run(debug=True)