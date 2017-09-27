# Milk Futures
Prediction of Milk futures movements

## Goal:  
Predict up or down movement of Milk Futures Prices

This information would give milk producers an idea of whether or not they should increase production over the next week.  

I was able to get historical Milk Futures prices from Quandl.com.  Here I plot the prices, along with a rolling average.

https://github.com/FawcettB98/Milk_Futures/blob/master/images/rolling_mean.png

### Baseline Prediction  

I will consider my model to be successful if it can predict the movement of the futures prices with more accuracy than a very simple model where you predict that in this period the model will move in the same direction as it did in the last period.  I ran a test with this strategy and found that it predicted the movements with an accuracy of 57.7%.  This is actually quite good, so will be tough to beat.

## Time series
### Random Walk

Most financial instrument prices follow what is known as a random walk.  For our purposes this means that it is difficult or impossible to predict their movements. Another term for this in time-series lingo is that the series is "stationary".   There is a statistical test for stationarity known as the Dicky-Fuller Test.  I ran the Dicky Fuller test on the Milk Futures time series and came up with a p-value of 0.24.  This can be interpreted as showing that the series is not stationary, so there may be some signal to pull out of the series.

### ARIMA

We'll start with trying to make a prediction based solely on the past prices of the Milk Futures.  To do this we will attempt a standard Time Series model called the Autoregressive Integrated Moving Average.  This model requires 3 parameters:  One for the Autoregressive part, one for the Integrated part and one for the Moving Average part.  One way to determine these parameters is to look at how the values of the series correlate to the values at earlier periods using Auto Correlation Plots:

*Plot ACF/PACF*

Without going into a lot of detail, these plots indicate that an AR parameter of 9 and an MA parameter of 0 seem appropriate.  We will also use an I parameter of 1 to make the series more stationary.  Let's look at how well the model fits the data, and at some predictions:

*Plot Fit/Predictions*

We are interested in looking at the up or down movement of the market and how well the model predicts this.  In this case, the accuracy is only 50%, so this model does not outperform our baseline strategy.

### ARIMAX
It is possible that other items impact the price of milk.  For example, the amount of Milk produced in the nation.  I was able to find monthly historical data on milk production from NASS.gov.  I also used Quandl.com to find additional futures prices of commodities that might be related to milk:  cheese, dry milk, and corn.

A model similar to the ARIMA model that allows for additional predictors is the ARIMAX model.  Here we show the fit and predictions of the ARIMAX model:

*Plot Fit/Predictions*

The accuracy of this model is 57.9%, so very slightly better than the baseline prediction.


## Neural network
Another way to approach time series is with what is known as a Recurrent Neural Net.  These also have parameters that need to be set, but there is not a simple plot to help, so I had to do some searching.  Each time a neural net is run the output changes slightly because of random inputs.  As such, I looked at the distribution of the accuracy for different values of the parameters.  In particular, I searched among different values for the Epochs, Batch Size and number of Neurons.

*Plot Boxplots (epochs, batch_size, neurons)*

Looking at these box-plots, it looks like the best parameters to use are 1000 epochs, a batch size of 50, and 100 neurons.  

*Plot rnn_act_v_pred*

The final average accuracy turns out to be about 63%.  This is significantly better than the baseline accuracy.
