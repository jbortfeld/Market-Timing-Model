import pandas as pd
import numpy as np
from pandas_datareader import data
import fix_yahoo_finance as yf
import statsmodels.api as sm
import statsmodels


yf.pdr_override()


def get_stock_data(tickers=['SPY'],
                   start_date='2018-01-01',
                   end_date='2018-05-31',
                   periodicity='monthly'):
    '''
    get end-of-day stock price and return information

    :param tickers: stock tickers to download, as list of strings
    :param data_source: which data service to use (currently only works with google), as string
    :param start_date: start date of data in "YYYY-MM-DD" format, as string
    :param start_date: end date of data in "YYYY-MM-DD" format, as string
    :return: price and volume data for given tickers in stacked format, as dataframe
    '''

    assert isinstance(tickers, list), 'error: tickers must be a list'
    assert isinstance(start_date, str), 'error: start_date must be a string'
    assert isinstance(end_date, str), 'error: end_date must be a string'
    assert periodicity in ['daily', 'monthly'], 'error: invalid periodicity'

    # download data
    df = data.get_data_yahoo(tickers, start_date, end_date)

    # if monthly periodicity, only keep month-end dates
    if periodicity == 'monthly':
        df.reset_index(inplace=True, drop=False)
        df['month'] = df['Date'].dt.month
        mask = df['month'] != df['month'].shift(-1)
        df = df[mask]
        del df['month']
        df.set_index('Date', inplace=True)

    if len(tickers) == 1:
        # if only one ticker, then ticker name is not in the dataframe
        # add it
        df['ticker'] = tickers[0]

    else:
        # if multiple tickers, then the data uses multiindex colum format
        # stack data to get the data for ticker1, then the data for ticker2 underneath it, etc
        df = df.stack()

    # rename columns
    df.reset_index(inplace=True, drop=False)
    df.rename(columns={'Date': 'date',
                       'level_1': 'ticker',
                       'Close': 'price'}, inplace=True)
    df.sort_values(by=['ticker', 'date'], inplace=True)

    # calculate daily return by ticker
    df['return'] = df.groupby(by='ticker')['price'].apply(lambda x: x / x.shift(1) - 1.0)

    return df[['ticker', 'date', 'price', 'return']]

def WLS_regression(data, 
                   x_vars = ['industrial_production', 'change_inflation', 'credit_risk_premium',
           'slope_interest_rate', 'housing_starts', 'delinquencies', 'change_unemployment'],
                   rho = 0.99):
    
    df = data.copy()
    df.dropna(axis = 0, inplace = True)
    
    # make sure the data is sorted chronologically
    df.sort_values(by = 'portfolio_date', inplace = True)
    
    # get the number of observations in the dataset
    big_t = df.shape[0] + 1
    
    # construct the weights to use for each observation
    # the most distant observation will have a small weight
    # and the most recent observation will have a big weight
    weights = []
    for small_t in range(1, big_t):
        weights.append(rho**(big_t - small_t))
    weights = np.array(weights)
    
    # create the explanatory variables
    X = df[x_vars]
    
    # now fit a model using the statsmodel WLS function
    #sm.WLS(y_data, x_data, weights)
    model_wls = sm.WLS(df['forward_spy_return'], statsmodels.tools.tools.add_constant(X), weights = weights)
    fit_wls = model_wls.fit()
    
    # save the coefficients into a dictionary
    results = fit_wls.params.to_dict()
    
    # add the r-squared, number of observations, etc
    results['r_squared'] = fit_wls.rsquared
    results['r_squared_adjusted'] = fit_wls.rsquared_adj

    results['n_obs'] = fit_wls.nobs
    results['mse'] = fit_wls.mse_total
    results['aic'] = fit_wls.aic
    results['model_vars'] = list(fit_wls.params.to_dict().keys())
    
    # add the pvalues (statistical significance of each coefficient)
    pvalues_dict = fit_wls.pvalues.to_dict() 
    for p in pvalues_dict.keys():
        results['{}_pval'.format(p)] = pvalues_dict[p]
        
    # add the date of the estimation
    results['portfolio_date'] = df['portfolio_date'].max()
    
    return results