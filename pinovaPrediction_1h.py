from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    n_features = ts.shape[1]
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

skenderovci = pd.read_csv('data/skenedrovci.csv')
skenderovci['readable_time'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in skenderovci['readable_time']]
skenderovci.sort_values('readable_time', inplace=True)
skenderovci['date'] = [x.date() for x in skenderovci['readable_time']]
skenderovci['hour'] = [x.hour for x in skenderovci['readable_time']]
skenderovci['month'] = [x.month for x in skenderovci['readable_time']]

# Hiperparametri modela
lag = 24
n_ahead = 1
test_share = 0.1
epochs = 100
batch_size = 128
lr = 0.001
n_layer = 50


drop_c = ['time', 'station_id']
cols = ['readable_time', 'station_id', 'air_temp', 'air_moisture', 'global_radiation', 'ground_moisture', 'leaf_temp',
        'leaf_wet', 'rainfall', 'wind_speed']

skenderovci.drop(labels=drop_c, axis=1, inplace=True)

features_considered = ['air_temp', 'air_moisture', 'global_radiation',
                       'leaf_wetness', 'rainfall', 'wind_speed']

ts = skenderovci[features_considered]

nrows = ts.shape[0]

# Podjela podataka
train = ts[0:int(nrows * (1 - test_share))]
test = ts[int(nrows * (1 - test_share)):]

# Skaliranje
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

ts_s = pd.concat([train, test])
ts_s = ts_s.apply (pd.to_numeric, errors='coerce')
ts_s = ts_s.dropna()
ts_s = ts_s.reset_index(drop=True)

X, Y = create_X_Y(ts_s.values, lag=lag, n_ahead=n_ahead)
n_ft = X.shape[2]

Xtrain, Ytrain = X[0:int(X.shape[0] * (1 - test_share))], Y[0:int(X.shape[0] * (1 - test_share))]
Xval, Yval = X[int(X.shape[0] * (1 - test_share)):], Y[int(X.shape[0] * (1 - test_share)):]

print(f"Shape of training data: {Xtrain.shape}")
print(f"Shape of the target data: {Ytrain.shape}")

print(f"Shape of validation data: {Xval.shape}")
print(f"Shape of the validation target data: {Yval.shape}")


class PredictionModel():
    def __init__( self, X, Y, n_outputs, n_lag, n_ft, n_layer, batch, epochs, lr, Xval=None, Yval=None,
                  min_delta=0.001, patience=10):
        lstm_input = Input(shape=(n_lag, n_ft))
        lstm_layer = LSTM(n_layer, activation='relu')(lstm_input)
        x = Dense(n_outputs)(lstm_layer)
        self.model = Model(inputs=lstm_input, outputs=x)
        self.batch = batch
        self.epochs = epochs
        self.n_layer = n_layer
        self.lr = lr
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.min_delta = min_delta
        self.patience = patience

    def trainCallback(self):
        return EarlyStopping(monitor='loss', patience=self.patience, min_delta=self.min_delta)
    def modelSave(self):
        return ModelCheckpoint('pinova_1korak_.h5', monitor='loss', mode='min', save_best_only=True)

    def train(self):
        empty_model = self.model
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        empty_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=optimizer)
        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.Xval, self.Yval),
                shuffle=False,
                callbacks=[self.trainCallback(), self.modelSave()])
        else:
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=False,
                callbacks=[self.trainCallback()])
        self.model = empty_model
        return history

    def predict(self, X):
        return self.model.predict(X)

model = PredictionModel(
    X=Xtrain,
    Y=Ytrain,
    n_outputs=n_ahead,
    n_lag=lag,
    n_ft=n_ft,
    n_layer=n_layer,
    batch=batch_size,
    epochs=epochs,
    lr=lr,
    Xval=Xval,
    Yval=Yval,
)
model.model.summary()

history = model.train()

loss = history.history.get('loss')
val_loss = history.history.get('val_loss')

n_epochs = range(len(loss))

plt.figure(figsize=(30, 15))
plt.plot(n_epochs, loss, 'r', label='Train_loss', color='blue')
if val_loss is not None:
    plt.plot(n_epochs, val_loss, 'r', label='Val_loss', color='red')
plt.legend(loc=0)
plt.xlabel('Epoha')
plt.ylabel('RMSE')
plt.show()

yhat = [x[0] for x in model.predict(Xval)]
y = [y[0] for y in Yval]

days = skenderovci['readable_time'].values[-len(y):]

frame = pd.concat([
    pd.DataFrame({'day': days, 'air_temp': y, 'type': 'original'}),
    pd.DataFrame({'day': days, 'air_temp': yhat, 'type': 'forecast'})
])

frame['air_temp_unscaled'] = [(x * train_std['air_temp']) + train_mean['air_temp'] for x in frame['air_temp']]

pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['air_temp_unscaled_original'] - pivoted['air_temp_unscaled_forecast']
pivoted['res_unsc'] = [pow(x,2) for x in pivoted['res']]


plt.figure(figsize=(30, 15))
plt.plot(pivoted.index, pivoted.air_temp_original, color='blue', label='original')
plt.plot(pivoted.index, pivoted.air_temp_forecast, color='red', label='forecast', alpha=0.6)
plt.title('Predviđanje temperature - skalirani podatci')
plt.legend()
plt.show()

plt.figure(figsize=(30, 15))
plt.plot(pivoted.index, pivoted.air_temp_unscaled_original - 276.15, color='blue', label='original')
plt.plot(pivoted.index, pivoted.air_temp_unscaled_forecast - 276.15, color='red', label='forecast', alpha=0.6)
plt.title('Predviđanje temperature')
plt.legend()
plt.show()

pivoted = frame.pivot_table(index='day', columns='type')
pivoted.columns = ['_'.join(x).strip() for x in pivoted.columns.values]
pivoted['res'] = pivoted['air_temp_unscaled_original'] - pivoted['air_temp_unscaled_forecast']
pivoted['res_mean'] = [pow(x,2) for x in pivoted['res']]


print(f"RMSE: {sqrt(pivoted['res_mean'].sum() / pivoted.shape[0])} C")

print(pivoted.res_mean.describe())
