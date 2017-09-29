from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import preprocessing
import matplotlib.pyplot as plt

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def rnn_multivariate(epochs=50, batch_size=72, neurons=50):

    # load dataset
    dataset = preprocessing.main()
    #dataset.drop(['Value'], axis=1, inplace=True)
    values = dataset.values
    # integer encode direction
    #encoder = LabelEncoder()
    #values[:,4] = encoder.fit_transform(values[:,4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x = scaler_x.fit_transform(values[:, :-1])
    scaled_y = scaler_y.fit_transform(values[:, -1].reshape(-1,1))
    scaled = concatenate((scaled_x, scaled_y), axis=1)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    #print reframed.head()
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
    #print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train_hours = 52*3
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    #print history
    # pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(history.history['val_loss'], label='test')
    # pyplot.legend()
    # pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    #inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler_y.inverse_transform(yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    # inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler_y.inverse_transform(test_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    fig, ax = plt.subplots()
    ax.plot(test_y, label='Actual')
    ax.plot(yhat, label='Predicted', c='r')
    ax.plot(yhat+1.96*rmse, '--', c='r', alpha=.5)
    ax.plot(yhat-1.96*rmse, '--', c='r', alpha=.5)
    #ax.set_xticklabels(dataset.index[-42:].date, rotation=45)
    #pyplot.fill_between(yhat + rmse, yhat - rmse)
    pyplot.ylabel('Normalized Milk Futures Prices')
    pyplot.xlabel('Week')
    pyplot.legend()
    pyplot.savefig('rnn_act_v_pred.png')
    pyplot.show()


    test_output = pd.concat([pd.DataFrame(test_y), pd.DataFrame(yhat)], axis=1, keys=['actual','predicted'])

    test_output['actual_chg'] = test_output.actual.diff().fillna(0).values > 0
    test_output['predicted_chg'] = test_output.predicted.diff().fillna(0).values > 0
    test_output['comparison'] = test_output.actual_chg == test_output.predicted_chg

    acc = accuracy_score(test_output['actual_chg'], test_output['predicted_chg'])
    print('Accuracy: %.3f' % acc)
    return acc

def experiment(repeats, epochs, batch_size, neurons):
    '''

    '''

    acc_scores = []
    for i in xrange(repeats):
        acc_scores.append(rnn_multivariate(epochs=epochs, batch_size=batch_size, neurons=neurons))
    return acc_scores

def run_experiment():
    '''

    '''

    results = pd.DataFrame()
    repeats = 10
    epochs = [50, 100, 500, 1000]
    batch = [20, 50, 70, 100]
    neurons = [25, 50, 100]
    for e in epochs:
        results[str(e)] = experiment(repeats, e, 50, 100)
    print results.describe()
    results.boxplot()
    plt.title('Number of Epochs')
    plt.savefig('boxplot_epochs.png')
    plt.show()

if __name__ == '__main__':
    rnn_multivariate(epochs=1000, batch_size=50, neurons=50)
#    run_experiment()
