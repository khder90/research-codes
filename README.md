# research-codes
(CNN, ConvLSTM, and Fourier-CNN) for Rainfall
CNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

data = pd.read_excel(r'C:\Users\khder\Desktop\RAIN.xlsx')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
rainfall = data['RainFall'].values.reshape(-1, 1)

scaler = MinMaxScaler()
rainfall_scaled = scaler.fit_transform(rainfall)

train_size = int(len(rainfall_scaled) * 0.7)
val_size = int(len(rainfall_scaled) * 0.15)
test_size = len(rainfall_scaled) - train_size - val_size

train = rainfall_scaled[:train_size]
val = rainfall_scaled[train_size:train_size+val_size]
test = rainfall_scaled[train_size+val_size:]

def create_dataset(dataset, look_back=12):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 12
X_train, y_train = create_dataset(train, look_back)
X_val, y_val = create_dataset(val, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()

checkpoint_path = "cnn_model.keras"
early_stop = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint],
    verbose=1
)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("cnn_training_plot.png")
plt.show()

model.load_weights(checkpoint_path)

def evaluate_model(X, y, name):
    y_pred = model.predict(X)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_true_inv = scaler.inverse_transform(y.reshape(-1, 1))
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)
    print(f"{name} - MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}, R2: {r2}")

evaluate_model(X_train, y_train, "Train")
evaluate_model(X_val, y_val, "Validation")
evaluate_model(X_test, y_test, "Test")

# Step 5: Forecast next 48 months
last_sequence = rainfall_scaled[-look_back:].reshape(1, look_back, 1)
future_preds = []

for _ in range(48):
    next_pred = model.predict(last_sequence)
    future_preds.append(next_pred[0][0])
    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred[0][0]]]], axis=1)

future_preds = np.array(future_preds).reshape(-1, 1)
future_preds_inv = scaler.inverse_transform(future_preds)

for i, value in enumerate(future_preds_inv.flatten(), 1):
    print(f"Month {i}: {value}")



ConvLSTM

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
file_path = r'C:\Users\khder\Desktop\RAIN.xlsx'
data = pd.read_excel(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
rainfall = data['RainFall'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
rainfall_scaled = scaler.fit_transform(rainfall)

n_total = len(rainfall_scaled)
n_train = int(n_total * 0.7)
n_test = int(n_total * 0.2)
n_val = n_total - n_train - n_test

train = rainfall_scaled[:n_train]
test = rainfall_scaled[n_train:n_train + n_test]
val = rainfall_scaled[n_train + n_test:]

def create_sequences(data, n_steps=12):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 12
X_train, y_train = create_sequences(train, n_steps)
X_test, y_test = create_sequences(test, n_steps)
X_val, y_val = create_sequences(val, n_steps)

# Reshape for ConvLSTM2D
def reshape_for_convlstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1, 1, 1))

X_train = reshape_for_convlstm(X_train)
X_test = reshape_for_convlstm(X_test)
X_val = reshape_for_convlstm(X_val)

def create_sequences(data, n_steps=12):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 12
X_train, y_train = create_sequences(train, n_steps)
X_test, y_test = create_sequences(test, n_steps)
X_val, y_val = create_sequences(val, n_steps)

def reshape_for_convlstm(X):
    return X.reshape((X.shape[0], X.shape[1], 1, 1, 1))

X_train = reshape_for_convlstm(X_train)
X_test = reshape_for_convlstm(X_test)
X_val = reshape_for_convlstm(X_val)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, ConvLSTM2D, BatchNormalization, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model = Sequential([
    Input(shape=(n_steps, 1, 1, 1)),
    ConvLSTM2D(filters=32, kernel_size=(1, 1), activation='tanh', return_sequences=False),
    BatchNormalization(),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_val, y_val),
                    callbacks=[early_stop, reduce_lr], verbose=0)

import matplotlib.pyplot as plt
import os

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ConvLSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.expanduser(r'~/Desktop/ConvLSTM_Training_Loss.png'))
plt.close()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
val_pred = model.predict(X_val)

train_pred_inv = scaler.inverse_transform(train_pred)
test_pred_inv = scaler.inverse_transform(test_pred)
val_pred_inv = scaler.inverse_transform(val_pred)

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

train_metrics = evaluate(y_train_inv, train_pred_inv)
test_metrics = evaluate(y_test_inv, test_pred_inv)
val_metrics = evaluate(y_val_inv, val_pred_inv)

print("Train:", train_metrics)
print("Test:", test_metrics)
print("Validation:", val_metrics)

future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1), periods=48, freq='M')
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Rainfall': future_preds.flatten()})
forecast_df.to_excel(r'C:\Users\khder\Desktop\Future_48_Months_Predictions.xlsx', index=False)
print(forecast_df)

CNN- Fourier 

scaler = MinMaxScaler()
rain_scaled = scaler.fit_transform(rain)

rain_fft = fft(rain_scaled.flatten())
rain_fft_filtered = np.copy(rain_fft)
rain_fft_filtered[30:] = 0
rain_denoised = np.real(ifft(rain_fft_filtered)).reshape(-1, 1)

def create_dataset(data, window_size=12):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


window_size = 12
X, y = create_dataset(rain_denoised, window_size)

train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
model = Sequential([
    Conv1D(64, kernel_size=2, activation='relu', input_shape=(window_size, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(X_train_cnn, y_train, validation_data=(X_val_cnn, y_val), epochs=100, batch_size=16, verbose=0)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig("training_validation_loss.png")
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


y_train_pred = model.predict(X_train_cnn)
y_val_pred = model.predict(X_val_cnn)
y_test_pred = model.predict(X_test_cnn)


y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_train_pred_inv = scaler.inverse_transform(y_train_pred)
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))
y_val_pred_inv = scaler.inverse_transform(y_val_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_inv = scaler.inverse_transform(y_test_pred)


    return {
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

print("Training Performance:", evaluate(y_train_inv, y_train_pred_inv))
print("Validation Performance:", evaluate(y_val_inv, y_val_pred_inv))
print("Testing Performance:", evaluate(y_test_inv, y_test_pred_inv))

forecast_input = X[-1].reshape(1, window_size, 1)
forecast_scaled = []

for _ in range(48):
    pred = model.predict(forecast_input)[0][0]
    forecast_scaled.append(pred)
    forecast_input = np.roll(forecast_input, -1)
    forecast_input[0, -1, 0] = pred

forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
