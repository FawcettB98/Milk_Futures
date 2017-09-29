# Milk Futures
Prediction of Milk futures movements

## What are futures contracts?

A futures contract is an agreement to buy or sell commodities at a fixed price to be delivered on a future date.  These contracts are then bought and sold on organized exchanges.

## Goal:  
Predict up or down movement of Milk Futures Prices

This information would give milk producers an idea of whether or not they should increase production over the next week.  

## Data Sources

Quandl.com - API with historic futures prices for Milk, Cheese, Dry Milk and corn

Weekly data  Start: 12/01/2013  End: 9/24/2017

NASS.usda.gov - National Agricultural Statistics Services QuickStats table with monthly milk production https://quickstats.nass.usda.gov/results/2F655051-7BEE-3F05-BA0F-0EA083C6F1F5

Monthly data  Start 01/2011  End 08/2017

![Milk Futures Prices](https://github.com/FawcettB98/Milk_Futures/blob/master/images/rolling_mean.png)

Historical prices for Milk Futures along with moving average Mean and Standard Deviation.

## Baseline Prediction  

The model will be considered successful if it can predict the movement of the futures prices with more accuracy than a very simple strategy. This strategy will be to  predict that the price will move in the same direction as it did in the previous period.  A test was run with this strategy and found that it predicted the movements with an accuracy of 57.7%.  

## Time series
Most financial instrument prices can be modeled as time series since the time component is key.  In order to be modeled a time series must be "stationary", which means that the statistical properties (mean, variance, autocorrelation, etc) are constant over time.   There is a statistical test for stationarity known as the Dicky-Fuller Test.  For the Milk Futures time series this test came up with a p-value of 0.24.  This can be interpreted as showing that the series is not stationary.  There are ways to adjust for this, such as differencing the data, which are included in the methods described below.

### ARIMA

The first model is based solely on the past prices of the Milk Futures.  This is called an Autoregressive Integrated Moving Average (ARIMA).  This model requires 3 parameters:  One for the Autoregressive part (how many periods of past values to look at), one for the Integrated part (how many times to difference the data) and one for the Moving Average part (how many periods of past errors to look at).  One way to determine these parameters is to look at how the values of the series correlate to the values at earlier periods using Auto Correlation Plots:

![ACF Plot](https://github.com/FawcettB98/Milk_Futures/blob/master/images/acf.png)

![PACF Plot](https://github.com/FawcettB98/Milk_Futures/blob/master/images/pacf.png)

These plots indicate that an AR parameter of 9 (look back 9 periods at the values) and an MA parameter of 0 (no Moving Average component) seem appropriate.  An I parameter of 1 was used to make the series more stationary.  The data was split into a "training" dataset of 156 weeks and a "test" dataset of 38 weeks.  The following graphs show how well the model fits the training data along with predictions into the test data:

![Model Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arima_fit.png)

![Model Prediction](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arima_pred.png)

There is a lot of variance in the predictions, so making predictions about the exact value of the futures price is difficult.  The desired metric, however, is the up or down movement of the market and how well the model predicts this.  In this case, the accuracy is only 50%, so this model does not outperform our baseline strategy.

### ARIMAX
It is possible that other items impact the price of milk.  For example, the amount of Milk produced in the nation.  Milk Production data was found at www.nass.usda.gov.  Additional futures prices of commodities that might be related to milk were also considered:  cheese, dry milk, and corn.

A model similar to the ARIMA model that allows for additional predictors is the ARIMAX model (the X stands for "Exogenous Variables", or Additional Variables).  The fit and predictions of the ARIMAX model are shown below:

![Model Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arimax_fit.png)

![Model Prediction](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arimax_pred.png)

The accuracy of this model is 57.9%, so very slightly better than the baseline prediction.


## Neural network
A competing model was created using a Recurrent Neural Networks.  

![Boxplot Epochs](https://github.com/FawcettB98/Milk_Futures/blob/master/images/deep_neural_network.png)

Training occurs in the hidden (blue) layers by minimizing error.  The particular type of network used - Long Short-Term Memory (LSTM) - allows the network to look into the past in the training.

There are several parameters that need to be "tuned" in the network.  These include:

Epoch:  "one pass over the entire dataset".  Used to separate the trianing into distinct phases.

Batch Size:  The number of samples that are propagated through the network

Neurons:  Number of interconnected units in the network

Since neural networks use random inputs, the results for each run are slightly different.  Multiple runs were used for various levels of these parameters.  Boxplots of the accuracy for the various parameters are shown below:

![Boxplot Epochs](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_epochs.png)

![Boxplot Batch Size](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_batchsize.png)

![Boxplot Neurons](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_neurons.png)

The boxplot for the Epochs shows a clear improvement as the number of epochs increases.  The final number of epochs chosen was 1000.

The boxplot for Batch Size does not show a clearly superior parameter setting.  The final Batch Size was set to 50.

The boxplot for Neurons does not show a clearly superior parameter setting.  The final Batch Size was set to 100.

![RNN Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/rnn_act_v_pred.png)

This plot shows the actual prices against the predicted prices, along with a margin of error of one standard deviation for the predictions.

The y-axis is the normalized price (converted to a number between 1 and 0), and the x-axis is the number of weeks since 12-04-2016

The final average accuracy turns out to be about 63%.  This is roughly 10% better than the baseline accuracy.

## Conclusion
Market prices are notoriously difficult to predict.  The use of machine learning models like LSTM Networks can help give a prediction of the future movement of the prices.

## Next Steps
Additional steps include finding more variables that may help predict the movement of Milk Futures prices.  Among these are feed prices and weather data.  
