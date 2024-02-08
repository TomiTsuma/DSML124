import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import os
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense,Dropout, LSTM, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn, optim
import torch.nn.functional as F 
import copy
import pickle


def run_lstm(chemical, n_epochs):
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(f'outputFiles/PCC1/train/{chemical}.csv', index_col=0, engine='c')
    sequences = df.astype(np.float32).to_numpy().tolist()
    sequences = sequences
    sequences = [normalize(i) for i in sequences]
    train_dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    val_dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, n_features = torch.stack(train_dataset).shape

    model = RecurrentAutoencoder(seq_len, n_features, 128)
    model = model.to(device)

    n_epochs = n_epochs


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
  
    for epoch in range(1, n_epochs + 1):
        print("Epoch, {}".format(epoch))
        model = model.train()
        train_losses = []
        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    pd.Series(train_losses).to_csv("losses.csv")
    print("Done")
    # #Next We need to evaluate the dataset
    # val_losses = []
    # best_losses = []
    # model = model.eval()
    # with torch.no_grad():
    #   for seq_true in val_dataset:
    #     seq_true = seq_true.to(device)
    #     seq_pred = model(seq_true)
    #     loss = criterion(seq_pred, seq_true)
    #     val_losses.append(loss.item())
    #     train_loss = np.mean(train_losses)
    #     val_loss = np.mean(val_losses)
    #     history['train'].append(train_loss)
    #     history['val'].append(val_loss)
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         best_model_wts = copy.deepcopy(model.state_dict())
    # print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "model/{}.torch".format(chemical) )
    pickle.dump(model, open('model/{}.pkl'.format(chemical), 'wb'))
    print(model)
    return model





class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
        input_size=n_features,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=embedding_dim,
        num_layers=1,
        batch_first=True
        )
    def forward(self, x):
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.detach().numpy().reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(torch.tensor(x.tolist()).to(device))
        x, (hidden_n, _) = self.rnn2(torch.tensor(x.tolist()).to(device))
        return hidden_n.reshape((self.n_features, self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
            )
        #Using a dense layer as an output layer
        self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
    def forward(self, x):
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.repeat(self.seq_len, self.n_features)
        x = x.detach().numpy().reshape((self.n_features, self.seq_len, self.input_dim))
        x, (hidden_n, cell_n) = self.rnn1(torch.tensor(x.tolist()).to(device))
        x, (hidden_n, cell_n) = self.rnn2(torch.tensor(x.tolist()).to(device))
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x













def run(chemical):
    # if(f"{chemical}" in os.listdir("outputFiles/models")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC1")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC3")):
    #     return
    # if(f"{chemical}.csv" not in os.listdir("outputFiles/PCC2")):
    #     return
    df = pd.read_csv(f'outputFiles/PCC1/train/{chemical}.csv', index_col=0, engine='c')
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


    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mae')
    history = autoencoder.fit(X_spc, X_spc, 
          epochs=200, 
          batch_size=32,
          validation_data=(X_spc, X_spc),
          shuffle=True)

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
    
def run_lstm_autoencoder(chemical):
    df = pd.read_csv(f'outputFiles/PCC1/train/{chemical}.csv', index_col=0, engine='c')
    model = lstm_autoencoder(np.array(df))
    os.makedirs(f'models/{chemical}', exist_ok=True)
    model.save(f'models/{chemical}', save_format='tf')

    return 

def lstm_autoencoder(X):
    callbacks = []
    earlystopping = EarlyStopping(monitor="val_loss",
                                                     patience=75,
                                                     min_delta=0,
                                                     mode='min',
                                                     restore_best_weights=False,
                                                     baseline=None,
                                                     verbose=0)
    callbacks.append(earlystopping)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.5,
                                                     patience=10,
                                                     min_lr=0.000001,
                                                     cooldown=5)

    callbacks.append(reduce_lr)

    n_in = len(X[0])
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    optimizer = Adam(clipvalue=3, learning_rate=0.00001, )
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X,X,
              epochs=2, 
          validation_data=(X, X),
          callbacks=callbacks,
          shuffle=True)


    return model
    # model.predict(X)


