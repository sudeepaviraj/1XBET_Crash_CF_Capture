import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask


# Load the dataset
data = pd.read_csv('data.csv')

data.columns=["flight","crash","timestamp"]

def train():
    # Visualize the dataset
    plt.plot(data['flight'], data['crash'])
    plt.xlabel('Time')
    plt.ylabel('crash')
    plt.title('Time Series Data')
    plt.show()

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['crash'] = scaler.fit_transform(data[['crash']])

    # Create sequences
    def create_sequences(data, time_steps=10):
        x = []
        y = []
        for i in range(len(data) - time_steps):
            x.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(x), np.array(y)

    time_steps = 10
    values = data['crash'].values.reshape(-1, 1)
    x_sequences, y_sequences = create_sequences(values, time_steps)

    # Reshape for RNN input
    x_sequences = x_sequences.reshape((x_sequences.shape[0], x_sequences.shape[1], 1))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_steps, 1)),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')


    # Train the model
    model.fit(x_sequences, y_sequences, epochs=10000)

    # Generate predictions
    x_test = values[-time_steps:].reshape((1, time_steps, 1))  # Use the last 'time_steps' values for prediction
    y_pred = model.predict(x_test)

    # Inverse transform the prediction to get the original scale
    y_pred_original = scaler.inverse_transform(y_pred)

    print(f"Predicted next value: {y_pred_original.flatten()[0]}")

    predictions = model.predict(x_sequences)

    predictions_original = scaler.inverse_transform(predictions)
    y_sequences_original = scaler.inverse_transform(y_sequences.reshape(-1, 1))

    # Plot the real values and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(data['flight'][time_steps:], y_sequences_original, label='Real Value')
    plt.plot(data['flight'][time_steps:], predictions_original, label='Predicted Value', color='red')
    plt.xlabel('flight')
    plt.ylabel('Value')
    plt.title('Real vs Predicted Values')
    plt.legend()
    plt.show()
    return model,data,scaler

def predict(model,data,scaler):
    # scaler = MinMaxScaler(feature_range=(0, 1))
    data = pd.read_csv('data.csv')

    data.columns=["flight","crash","timestamp"]
    time_steps = 3
    last_sequence = data['crash'].values[-time_steps:].reshape((1, time_steps, 1))

    # Predict the next value
    predicted_value = model.predict(last_sequence)

    # Inverse transform the predicted value to get the original scale
    predicted_value_original = scaler.inverse_transform(predicted_value)

    print(f"Predicted next value: {predicted_value_original.flatten()[0]}")

model,data,scaler = train()
        
app = Flask(__name__)

@app.route("/")
def hello_world():
    predict(model,data,scaler)

app.run(host='0.0.0.0',port='5000')