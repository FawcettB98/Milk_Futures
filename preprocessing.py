import pandas as pd
import numpy as np
import quandl
import os
import matplotlib.pyplot as plt
import calendar
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

api_code = os.environ["QUANDL_ACCESS_KEY_ID"]
quandl.ApiConfig.api_key = api_code

def read_data(quandl_code, \
              column='Settle', \
              start_date='2013-12-01', \
              end_date='2017-09-24', \
              offset=0):

    '''
    Reads data from Quandl.com.  Must have an active api code.

    Input:
        quandle_code:  str - code indicating which security/derivative you want
            information on
        column:  str - the column of the data you want to pull
        start_date:  str
        end_date:  str
        offset:  int - how many periods to offset the data

    Output:
        data:  pandas DataFrame with one column
    '''
    data = quandl.get(quandl_code, \
                      start_date=start_date, \
                      end_date=end_date, \
                      collapse='weekly')

    data = data[column].shift(offset).fillna(0)

    return data

def combine_data(datasets, names):
    '''
    Combines the data into one dataframe, including data from the NASS on milk
        production (this data needs to be in a csv file in the same directory)

    Input:
        datasets:  list[pandas Dataframes] - the set of datasets to be combine_data
        names:  list[strings] - names of the columns in output object

    Output:
        data3:  pandas DataFrame
    '''

    data = pd.DataFrame()
    for i, dataset in enumerate(datasets):
        data[names[i]] = dataset

    data['month'] = data.index.month
    data['year'] = data.index.year
    data['str_m'] = data.month.apply(lambda x: '0' + str(x) if len(str(x)) < 2 else str(x))
    data['str_y'] = data.year.apply(lambda x: str(x))

    data['ym'] = data.str_y + "-" + data.str_m
    data.drop(['year','month','str_m','str_y'], axis=1, inplace=True)

    production = pd.read_csv('milk_production.csv', thousands=',')
    production_value = production[['Year', 'Period', 'Value']]

    scaler = MinMaxScaler()
    production_value['Value'] = scaler.fit_transform(production_value['Value'].values.reshape(-1,1))

    month_dict = dict((v,k) for k,v in enumerate(calendar.month_abbr))
    production_value['Month'] = production_value['Period'].apply(lambda x: month_dict[x.capitalize()])

    production_value['str_m'] = production_value.Month.apply(lambda x: '0' + str(x) if len(str(x)) < 2 else str(x))
    production_value['str_y'] = production_value.Year.apply(lambda x: str(x))

    production_value['ym'] = production_value.str_y + "-" + production_value.str_m
    production_value.drop(['Year', 'Period', 'Month', 'str_m', 'str_y'], axis=1, inplace=True)

    data3 = data.merge(production_value,on='ym', how='left')
    data3.set_index(data.index, inplace=True)
    data3['Value'] = data3['Value'].shift(-1)
    data3.drop('ym', axis=1, inplace=True)
    data3.fillna(0, inplace=True)

    return data3

def base_accuracy_score(dataset, column):
    '''
    Calculates the baseline accuracy assuming the movement in the price is the
        same as the week before

    Inputs:
        dataset:  pandas DataFrame - contains the column to be analyzed.
        column:  str - the column to be analyzed

    Output:
        base_predictions:  pandas DataFrame - contains two columns:
                                                Actual price movement
                                                Predicted price movement
    '''
    actual_chg = []
    for i in xrange(dataset.shape[0]-1):
        actual_chg.append(dataset[column][i] < dataset[column][i+1])
        actual_chg_df = pd.DataFrame(actual_chg)

    base_pred = [False]
    for i in xrange(actual_chg_df.shape[0]-1):
        base_pred.append(actual_chg_df.iloc[i].values[0])
    base_pred_df = pd.DataFrame(base_pred)

    base_predictions = pd.concat([actual_chg_df, base_pred_df], axis=1, keys=('actual', 'predicted'))
    return base_predictions

def main():
    '''
    Reads data for milk, cheese, dry milk, and corn and combines them along with
        milk production data.

    Inputs:
        None

    Outputs:
        data:  pandas DataFrame
    '''

    milk = read_data('CHRIS/CME_DA1')
    cheese = read_data('CHRIS/CME_CSC1', offset=-1)
    dry = read_data('CHRIS/CME_NF1', offset=-1)
    corn = read_data('CHRIS/CME_C1', offset=-1)


    data = combine_data([milk, cheese, dry, corn], \
                        ['milk', 'cheese', 'dry', 'corn'])
    return data

if __name__ == '__main__':
    data = main()

    base_predictions = base_accuracy_score(data, 'milk')
    print accuracy_score(base_predictions['actual'], base_predictions['predicted'])
