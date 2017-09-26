# Milk_Futures
Prediction of Milk futures movements

## Goal:  
Predict up or down movement of Milk Futures Prices

Give milk producers an idea of whether or not they should increase production over the next week.

Plot series

####Baseline Prediction  
I will consider my model to be successful if it can predict the movement of the futures prices with more accuracy than a very simple model where you predict that in this period the model will move in the same direction as it did in the last period.  I ran a test with this strategy and found that it predicted the movements with an accuracy of 57.7%.  This is actually quite good, so will be tough to beat.

### Time series
####Random Walk
Most financial instrument prices follow what is known as a random walk.  For our purposes this means that it is difficult or impossible to predict their movements. Another term for this in time-series lingo is that the series is "stationary".   There is a statistical test for stationarity known as the Dicky-Fuller Test.  I ran the Dicky Fuller test on the Milk Futures time series and came up with a p-value of 0.24.  This can be interpreted as showing that the series is not stationary, so there may be some signal to pull out of the series.

####ARIMA


ARIMAX

### Neural network
LSTM
