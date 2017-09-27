# Milk Futures
Prediction of Milk futures movements

## Goal:  
Predict up or down movement of Milk Futures Prices

This information would give milk producers an idea of whether or not they should increase production over the next week.  

## Data Sources

Quandl.com - historic futures prices for Milk, Cheese, Dry Milk and corn
NASS.usda.gov - monthly milk production http://usda.mannlib.cornell.edu/MannUsda/viewDocumentInfo.do?documentID=1103


![Milk Futures Prices](https://github.com/FawcettB98/Milk_Futures/blob/master/images/rolling_mean.png)

## Baseline Prediction  

The model will be considered successful if it can predict the movement of the futures prices with more accuracy than a very simple model where you predict that in this period the model will move in the same direction as it did in the last period.  A test was run with this strategy and found that it predicted the movements with an accuracy of 57.7%.  

## Time series
### Random Walk

Most financial instrument prices can be modeled as time series since the time component is key.  In order to be modeled a time series must be "stationary", which means that the statistical properties (mean, variance, autocorrelation, etc) are constant over time.   There is a statistical test for stationarity known as the Dicky-Fuller Test.  For the Milk Futures time series this test came up with a p-value of 0.24.  This can be interpreted as showing that the series is not stationary.  There are ways to adjust for this, such as differencing the data, which are included in the methods described below.

### ARIMA

The first model is based solely on the past prices of the Milk Futures.  This is called an Autoregressive Integrated Moving Average (ARIMA).  This model requires 3 parameters:  One for the Autoregressive part (how many periods of past values to look at), one for the Integrated part (how many times to difference the data) and one for the Moving Average part (how many periods of past errors to look at).  One way to determine these parameters is to look at how the values of the series correlate to the values at earlier periods using Auto Correlation Plots:

![ACF Plot](https://github.com/FawcettB98/Milk_Futures/blob/master/images/acf.png)

![PACF Plot](https://github.com/FawcettB98/Milk_Futures/blob/master/images/pacf.png)

Without going into a lot of detail, these plots indicate that an AR parameter of 9 and an MA parameter of 0 seem appropriate.  We will also use an I parameter of 1 to make the series more stationary.  Let's look at how well the model fits the data, and at some predictions:

![Model Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arima_fit.png)

![Model Prediction](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arima_pred.png)

The desired metric is the up or down movement of the market and how well the model predicts this.  In this case, the accuracy is only 50%, so this model does not outperform our baseline strategy.

### ARIMAX
It is possible that other items impact the price of milk.  For example, the amount of Milk produced in the nation.  Milk Production data was found at www.nass.usda.gov.  Additional futures prices of commodities that might be related to milk were also considered:  cheese, dry milk, and corn.

A model similar to the ARIMA model that allows for additional predictors is the ARIMAX model (the X stands for "Exogenous Variables", or Additional Variables).  The fit and predictions of the ARIMAX model are shown below:

![Model Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arimax_fit.png)

![Model Prediction](https://github.com/FawcettB98/Milk_Futures/blob/master/images/arimax_pred.png)

The accuracy of this model is 57.9%, so very slightly better than the baseline prediction.


## Neural network
A competing model was created using a Recurrent Neural Net.  



![Boxplot Epochs](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_epochs.png)

![Boxplot Batch Size](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_batchsize.png)

![Boxplot Neurons](https://github.com/FawcettB98/Milk_Futures/blob/master/images/boxplot_neurons.png)

Looking at these box-plots, it looks like the best parameters to use are 1000 epochs, a batch size of 50, and 100 neurons.  

![RNN Fit](https://github.com/FawcettB98/Milk_Futures/blob/master/images/nrr_act_v_pred.png)

The final average accuracy turns out to be about 63%.  This is roughly 10% better than the baseline accuracy.
