from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import xgboost as xgb

app = Flask(__name__)

# Load the dataset from CSV file
data = pd.read_csv("sepsis dataset.csv")
data = data.drop(columns=['Gender'])
data[['Systolic', 'Diastolic']] = data['BloodPressure'].str.split('/', expand=True)
data['Systolic'] = data['Systolic'].astype(float)
data['Diastolic'] = data['Diastolic'].astype(float)
data = data.drop(columns=['BloodPressure'])

# Prepare the data for training
X = data.drop(columns=['Sepsis']).astype(float)
y = data['Sepsis'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=32)

# Define the RNN model creation function
def create_rnn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    return model

# Create and train the RNN model
rnn_model = create_rnn_model(input_shape=(X_train.shape[1],))
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Create and train the XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

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
    if request.method == 'POST':
        age = int(request.form['age'])
        heart_rate = int(request.form['heartRate'])
        temperature = float(request.form['temperature'])
        systolic, diastolic = map(float, request.form['bloodPressure'].split('/'))
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

        # LSTM model prediction
        lstm_prediction = lstm_model.predict(user_data.values.reshape(1, user_data.shape[1], 1))[0][0]

        # RNN model prediction
        rnn_prediction = rnn_model.predict(user_data)

        # XGBoost model prediction
        xgb_prediction = xgb_model.predict_proba(user_data)[0][1]

        # Ensemble prediction
        final_prediction = (lstm_prediction + rnn_prediction + xgb_prediction) / 3

        if final_prediction <= 0.5:
            return "No sepsis detected."
        else:
            return "Sepsis detected."

    return "Error: Invalid request method."

if __name__ == '__main__':
    app.run(debug=True)
