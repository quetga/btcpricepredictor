import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load the data
df = pd.read_csv("btc_price.csv")

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Define the number of timesteps and features
timesteps = 60
features = 1

# Create the training and testing datasets
def create_dataset(dataset, timesteps):
    X = []
    y = []
    for i in range(timesteps, len(dataset)):
        X.append(dataset[i-timesteps:i, 0])
        y.append(dataset[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], features))
    return X, y

X_train, y_train = create_dataset(train_data, timesteps)
X_test, y_test = create_dataset(test_data, timesteps)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the testing data
loss = model.evaluate(X_test, y_test)
print("Test loss:", loss)

# Make predictions on the testing data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot the predictions against the actual Bitcoin prices
import matplotlib.pyplot as plt
plt.plot(df["Date"][train_size+timesteps:], df["Close"][train_size+timesteps:])
plt.plot(df["Date"][train_size+timesteps:], predictions)
plt.legend(["Actual", "Predicted"])
plt.show()
