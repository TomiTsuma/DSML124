import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense,Dropout, LSTM, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run(chemical):
    # if(f"{chemical}" in os.listdir("outputFiles/models")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC1")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC3")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC2")):
    #     return
    df = pd.read_csv(f'outputFiles/PCC1/{chemical}.csv', index_col=0, engine='c')
    outlier_df = pd.read_csv(f'outputFiles/PCC3/{chemical}.csv', index_col=0, engine='c')

    outliers = np.array(outlier_df)
    X = np.array(df)
    
    if(len(X) < 1):
        return

    X_spc = []
    outlier_spc = []

    for i in range(len(X)):
        X_spc.append(normalize(X[i]))
    for spc in outliers:
        outlier_spc.append(normalize(spc))

    X_spc = np.array(X_spc)
    outlier_spc = np.array(outlier_spc)

    autoencoder = lstm_autoencoder(X_spc)

    # autoencoder = AnomalyDetector()
    # autoencoder.compile(optimizer='adam', loss='mae')
    # history = autoencoder.fit(X_spc, X_spc, 
    #       epochs=200, 
    #       batch_size=32,
    #       validation_data=(X_spc, X_spc),
    #       shuffle=True)

    # pred = autoencoder.predict(X_spc)
    # if(len(outliers) > 0):
    #     outlier_pred = autoencoder.predict(outlier_spc)

    # predictions = []
    # for reconstructed, orig_spc in zip(pred, X):
    #     min = (np.min(orig_spc))
    #     max = (np.max(orig_spc))
    #     predictions.append(denormalize(reconstructed, min, max))


    # if(len(outliers) > 0):
    #     outlier_predictions = []
    #     for reconstructed, orig_spc in zip(outlier_pred, outliers):
    #         min = (np.min(orig_spc))
    #         max = (np.max(orig_spc))
    #         outlier_predictions.append(denormalize(reconstructed, min, max))

    # mae = []
    # mse = []
    # for orig, reconstructed in zip(X, predictions):
    #     mae.append(mean_absolute_error(orig, reconstructed))
    #     mse.append(mean_squared_error(orig, reconstructed))


    # if(len(outliers) > 0):
    #     outlier_mae = []
    #     outlier_mse = []
    #     for orig, reconstructed in zip(outliers, outlier_predictions):
    #         outlier_mae.append(mean_absolute_error(orig, reconstructed))
    #         outlier_mse.append(mean_squared_error(orig, reconstructed))

    os.makedirs(f'outputFiles/models/{chemical}', exist_ok=True)
    autoencoder.save(f'outputFiles/models/{chemical}', save_format='tf')    
    return autoencoder

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data

def denormalize(normalized_data, original_min, original_max):
    denormalized_data = (normalized_data + 1) * (original_max - original_min) / 2 + original_min
    return denormalized_data

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      Dense(128, activation="relu"),
      Dense(64, activation="relu"),
      Dense(32, activation="relu"),
      Dense(16, activation="relu"),
      Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      Dense(16, activation="relu"),
      Dense(32, activation="relu"),
      Dense(64, activation="relu"),
      Dense(128, activation="relu"),
      Dense(256, activation="relu"),
      Dense(1728, activation="tanh")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
    


def lstm_autoencoder(X):
    n_in = len(X[0])
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mae')
    model.fit(X,X,
              epochs=20, 
          batch_size=32,
          validation_data=(X, X),
          shuffle=True)

    return model
    # model.predict(X)

