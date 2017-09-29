import pandas as pd
import numpy as np
import pyflux as pf
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import statsmodels.api as sm
import quandl
import requests
import os
import calendar
import preprocessing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score

def test_stationarity(timeseries):
    '''
    Plots and saves a rolling mean plot.
    Plots and saves an autocorrelation plot and a partial autocorrelation plot
    Prints the results of the Dicky-Fuller Test

    Inputs:
        timeseries:  pandas DataFrame - one column dataframe with the timeseries.

    Outputs:
        None
    '''

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig('rolling_mean.png')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(16, 4))
    _ = sm.graphics.tsa.plot_acf(timeseries, lags=2*52, ax=ax)
    plt.savefig('acf.png')
    plt.show()

    fig, ax = plt.subplots(1, figsize=(16, 4))
    _ = sm.graphics.tsa.plot_pacf(timeseries, lags=2*52, ax=ax)
    plt.savefig('pacf.png')
    plt.show()

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

def univariate_arima():
    '''
    Reads the data and fits the ARIMA model
    Prints the Acccuracy Score

    Inputs:
        None

    Outputs:
        None
    '''

    data = preprocessing.main()
    n_train_hours = 52*3
    train = data.iloc[:n_train_hours, :]
    test = data.iloc[n_train_hours:, :]

    model = pf.ARIMA(data=train,ar=9,ma=0,integ=1,target='milk')

    x = model.fit("MLE")
    x.summary()

    # model.plot_fit(figsize=(15,5))
    model.plot_predict(h=38,past_values=20,figsize=(15,5))
    #import pdb; pdb.set_trace()

    yhat = model.predict(h=38)
    pred_chg = yhat > 0
    actual_chg = test.iloc[:-1,0].diff() > 0
    print accuracy_score(actual_chg, pred_chg)


def multivariate_arima():
    '''
    Reads the data and fits the ARIMAX model
    Prints the Acccuracy Score

    Inputs:
        None

    Outputs:
        None
    '''

    data = preprocessing.main()
    n_train_hours = 52*3
    train = data.iloc[:n_train_hours, :]
    test = data.iloc[n_train_hours:, :]

    model = pf.ARIMAX(data=train, formula = 'milk~1+cheese+dry+corn+Value', \
                        ar=9, ma=0, integ=1)
    x = model.fit("MLE")
    x.summary()

    # model.plot_fit(figsize=(15,5))
    # model.plot_predict(h=38,past_values=20,figsize=(15,5), oos_data=test)

    yhat = model.predict(h=38, oos_data=test)
    pred_chg = yhat > 0
    actual_chg = test.iloc[:-1,0].diff() > 0
    print accuracy_score(actual_chg, pred_chg)


if __name__ == '__main__':
#    multivariate_arima()
    data = preprocessing.main()
    test_stationarity(data.milk)
