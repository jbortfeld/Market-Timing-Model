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
    results['llf'] = fit_wls.llf
    results['model_vars'] = list(fit_wls.params.to_dict().keys())
    
    # add the pvalues (statistical significance of each coefficient)
    pvalues_dict = fit_wls.pvalues.to_dict() 
    for p in pvalues_dict.keys():
        results['{}_pval'.format(p)] = pvalues_dict[p]
        
    # add the date of the estimation
    results['portfolio_date'] = df['portfolio_date'].max()
    
    return results

def WLS_regression_with_var_selection_r2(data,
                                         all_possible_vars,
                                         rho,
                                         verbose = False):

    # recall that high R^2 is good in contrast to low AIC being good
    # so set the initial R^2 performance really low
    current_performance = -999

    # get some training data
    temp = data.copy()

    # define the variables we want to use
    # initialize as empty list
    # (start with no variables since this is using forward variable selection)
    vars_to_use = []

    search = True

    iterations = 0
    while search:

        if verbose:
            print('iteration number: ', iterations)
        iterations += 1

        best_candidate_performance = -999.

        # iterate through variables
        for var in all_possible_vars:

            # if variable is not already in the list of variables we are using, then test adding it
            if not var in vars_to_use:

                vars_to_try = vars_to_use.copy()

                # add the candidate variable to the list of variables already in use
                vars_to_try.append(var)

                # estimate model
                results = WLS_regression(temp,
                                         x_vars = vars_to_try,
                                         rho = rho)

                # get the performance of this model
                # save as a tuple (performance, list of vars)
                performance = results['r_squared_adjusted']
                if verbose:
                    print('--adding {} results in {}'.format(var, performance))

                # if adding this variable improves the r^2, then consider adding it
                if performance > current_performance:

                    if performance > best_candidate_performance:
                        candidate_variable = var
                        best_candidate_performance = performance

        # if adding any variable doesn't increase model performance, then stop
        if best_candidate_performance < current_performance:
            search = False
            if verbose:
                print('break out: {}'.format(best_candidate_performance))

        # if adding a variable increases model performance, then do it
        else:
            vars_to_use.append(candidate_variable)
            current_performance = best_candidate_performance
        if verbose:
            print('done with iteration. Add {} and new best adjusted R^2 is {}'.format(candidate_variable, best_candidate_performance))
            print()

    if verbose:
        print('final variables to use:', vars_to_use)

    # now run the final model
    results = WLS_regression(temp,
                             x_vars = vars_to_use,
                             rho = rho)
    return results

def WLS_regression_with_var_selection_aic(data,
                                         all_possible_vars,
                                         rho,
                                         verbose = False):

    temp = data.copy()

    # get the performance (AIC) with all variables included in the model
    results = WLS_regression(temp, x_vars = all_possible_vars, rho=0.99)
    current_aic = results['aic']
    if verbose:
        print('starting AIC with all vars: ', current_aic)
        print()

    # define the variables we want to use
    # (again, starting with all variables)
    vars_to_use = all_possible_vars.copy()

    search = True

    iterations = 0
    while search:

        if verbose:
            print('iteration number: ', iterations)
        iterations += 1

        best_candidate_aic = 999999999.

        # iterate through variables
        for var in all_possible_vars:

            # if variable is in the list of variables we are using, then test removing it
            if var in vars_to_use:

                vars_to_try = vars_to_use.copy()

                # remove the candidate variable
                vars_to_try.remove(var)

                # estimate model
                results = WLS_regression(temp, x_vars=vars_to_try, rho=0.99)
                # get the performance of this model
                performance = results['aic']
                if verbose:
                    print('--removing {} results in {}'.format(var, performance))

                # if removing this variable improves the aic, then consider adding it
                if performance < current_aic:

                    if performance < best_candidate_aic:
                        candidate_variable = var
                        best_candidate_aic = performance

        # if removing a variable doesn't improve performance, then strop
        if best_candidate_aic > current_aic:
            search = False
            if verbose:
                print('break out: {}'.format(best_candidate_aic))

        # if removing a variable improves performance, then remove it and keep testing
        else:
            vars_to_use.remove(candidate_variable)
            current_aic = best_candidate_aic
        if verbose:
            print('done with iteration. Remove {} and new best aic is {}'.format(candidate_variable, current_aic))
            print()

    if verbose:
        print('final variables to use:', vars_to_use)

    # now run the final model
    results = WLS_regression(temp,
                             x_vars=vars_to_use,
                             rho=rho)
    return results
