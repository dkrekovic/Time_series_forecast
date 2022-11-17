from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Dropout

skenderovci = pd.read_csv('data/skenedrovci.csv', index_col='readable_time', parse_dates=True)

drop_c = ['time', 'station_id']
cols = ['readable_time', 'station_id', 'air_temp', 'air_moisture', 'global_radiation', 'ground_moisture', 'leaf_temp',
        'leaf_wet', 'rainfall', 'wind_speed']

skenderovci.drop(labels=drop_c, axis=1, inplace=True)

features_considered = ['air_temp', 'air_moisture', 'global_radiation',
                       'leaf_wetness', 'rainfall', 'wind_speed']
features = skenderovci[features_considered]


# pretvorba vremenske serije u problem nadziranog učenja
def series_to_supervised(data, n_in, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = features.values
values = values.astype('float32')
# normalizacija
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# broj lag-ova
n_hours = 24
n_features = 6

reframed = series_to_supervised(scaled, n_hours, 1)
print(reframed.shape)

# podjela u skup za treniranje i testiranje
values = reframed.values
n_train_hours = 511 * 24 #70%
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

n_obs = n_hours * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# pretvori ulaz u 3D[samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# modeliranje mreže
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
# rano zaustavljanje
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#mc= ModelCheckpoint('model_pin_24_.h5', monitor='loss', mode='min', save_best_only=True)
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


history = model.fit(train_X, train_y, epochs=100, batch_size=128, validation_data=(test_X, test_y), verbose=1,
                    shuffle=False, callbacks=[es,mc])

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val_loss')
pyplot.legend()
pyplot.show()



# predikcija
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
# invertiranje skaliranja za predikciju
inv_yhat = concatenate((yhat, test_X[:, -5:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invertiranje skaliranja za stvarnu vrijednost
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -5:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
#RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


pyplot.figure(figsize=(20, 10))
pyplot.plot(inv_y, color='green', linewidth=1, label='True value')
pyplot.plot(inv_yhat, color='blue', linewidth=1, label='Predicted')
pyplot.legend(frameon=False)
pyplot.ylabel("Temperatura zraka")
pyplot.title("Predviđanje temperature")
pyplot.show()

pyplot.figure(figsize=(20, 10))
pyplot.plot(inv_y[:-2160], color='green', linewidth=1, label='True value')
pyplot.plot(inv_yhat[:-2160], color='blue', label='Predicted')
pyplot.legend(frameon=False)
pyplot.ylabel("Temperatura zraka")
pyplot.title("Predviđanje zadnja 3 mjeseca")
pyplot.show()

pyplot.figure(figsize=(12, 8))
pyplot.scatter(inv_y, inv_yhat, s=2, color='blue')
pyplot.ylabel("Y true")
pyplot.xlabel("Y predicted")
pyplot.title("Scatter plot")
pyplot.show()


