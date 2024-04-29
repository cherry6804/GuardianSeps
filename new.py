import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Step 1: Load the dataset from CSV file
data = pd.read_csv("sepsis dataset.csv")

# Remove the 'Gender' column
data = data.drop(columns=['Gender'])

# Step 2: Preprocess the dataset
# Split the 'BloodPressure' column into 'Systolic' and 'Diastolic' columns
data[['Systolic', 'Diastolic']] = data['BloodPressure'].str.split('/', expand=True)

# Convert 'Systolic' and 'Diastolic' columns to float
data['Systolic'] = data['Systolic'].astype(float)
data['Diastolic'] = data['Diastolic'].astype(float)

# Drop the original 'BloodPressure' column
data = data.drop(columns=['BloodPressure'])

# Print preprocessed dataset
print("Preprocessed Dataset:")
print(data.head())

# Step 3: Define the LSTM model creation function
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    return model

# Step 4: Prepare the data for training
X = data.drop(columns=['Sepsis']).astype(float)
y = data['Sepsis'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the LSTM model
lstm_model = create_lstm_model(input_shape=(X_train.shape[1], 1))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=50, batch_size=32)

# Step 6: Define the RNN model creation function
def create_rnn_model(input_shape):
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),
        Dense(1, activation='sigmoid')
    ])
    return model

# Step 7: Train the RNN model
rnn_model = create_rnn_model(input_shape=(X_train.shape[1],))
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Step 8: Extract predictions from LSTM and RNN models
lstm_predictions = lstm_model.predict(X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1))
rnn_predictions = rnn_model.predict(X_test)

# Step 9: Combine predictions
combined_predictions = np.concatenate((lstm_predictions, rnn_predictions), axis=1)

# Step 10: Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(combined_predictions, y_test)

# Step 11: Ask for input from the user
def prompt_user_for_data():
    age = int(input("Enter age: "))
    heart_rate = int(input("Enter heart rate (bpm): "))
    temperature = float(input("Enter temperature (°C): "))
    blood_pressure = input("Enter blood pressure (systolic/diastolic): ")
    systolic, diastolic = map(float, blood_pressure.split('/'))
    respiratory_rate = int(input("Enter respiratory rate: "))
    white_blood_cell_count = int(input("Enter white blood cell count: "))
    lactic_acid = float(input("Enter lactic acid level: "))
    
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
    
    return user_data

user_data = prompt_user_for_data()
lstm_prediction = lstm_model.predict(user_data.values.reshape(1, user_data.shape[1], 1))
rnn_prediction = rnn_model.predict(user_data)
combined_prediction = np.concatenate((lstm_prediction, rnn_prediction), axis=1)
final_prediction = xgb_model.predict(combined_prediction)[0]

# Interpret final prediction
if final_prediction == 0:
    print("No sepsis detected.")
elif final_prediction == 1:
    print("Sepsis detected.")


