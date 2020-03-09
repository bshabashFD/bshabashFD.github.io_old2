#!/usr/bin/env python3
import sys
import re
import os
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import time


import logging



##################################################################
def attempt_getting_token(asset, attempt):
    '''
    A single attempt at getting a crumb authentication token.
    Sends a request to Yahoo finance and uses regex to find
    an instance of an authentication token.

    Parameters:
    ------------
    asset (str) -   the asset (stock code) whose page is visited on 
                    Yahoo Finance
    attempt (int) - the attempt number, how many times was this
                    action attempted

    Returns:
    -----------
    an authentication token (string) if successful, None otherwise
    '''
    logging.info(f'Attempt {attempt+1}: Getting token from Yahoo Finance')

    # Get the request and response
    yahoo_request = requests.get(f'https://ca.finance.yahoo.com/quote/{asset}/history?p={asset}')
    response_string = yahoo_request.content.decode("utf-8")

    # Search for our token (crumb store) with regex
    all_tokens = re.findall(r'"CrumbStore":{"crumb":"\w+"',response_string)
    if (len(all_tokens) > 0):
        # Replace everything that isn't the crumb with empty strings
        my_token_string = all_tokens[0].replace("\"CrumbStore\":{\"crumb\":", "").replace("\"", "")
    
        logging.info(f'Token found {my_token_string}')
        return my_token_string
    else:
        return None
##################################################################
def get_crumb_token(asset="%5EGSPC"):
    '''
    Attempts getting an authentication token 10 times using attempt_getting_crumb(asset, attempt)
    
    Parameters:
    ------------
    asset (string, default S&P500) -    the asset (stock code) whose page is visited on 
                                        Yahoo Finance

    Returns:
    -----------
    an authentication token (string) if successful, None otherwise
    '''
    logging.info('Attmepting to get token')
    my_token_string = None
    attempts = 0
    while (attempts < 10) and (my_token_string is None):
        
        # Try and get the authentication token
        my_token_string = attempt_getting_token(asset, attempts)

        attempts += 1

        # If unsuccessful, sleep so you don't bombard the website
        time.sleep(2)

    # Return the appropriate result and log the process
    if (my_token_string is None):
        logging.warning('Token not found')
        return None
    else:
        logging.info(f'Token found {my_token_string}')
        return my_token_string

##################################################################
def get_stock_data(STOCK_ID, start_time, end_time, auth_code):
    ''' 
    Gets the stock data for the request stock within the range
    specified by [start_time] and [end_time]

    Parameters:
    ------------
    STOCK_ID (string) - The asset for which data is requested
                        (e.g. AAPL, MSFT)
    start_time (int) -  The start time as an epoch time (e.g. 1420229214)
    end_time (int) -    The end time as an epoch time
    auth_code (string) -The authentication code used to make 
                        the request valid for Yahoo Finance

    Returns:
    -----------
    a pandas.core.DataFrame of the stock historic
    data if successful, None otherwise
    '''
    logging.info(f'Obtaining data for symbol {STOCK_ID}')

    # Grab the data
    request_string = f'https://query1.finance.yahoo.com/v7/finance/download/{STOCK_ID}?period1={start_time}&period2={end_time}&interval=1d&events=history&crumb={auth_token}'
    r = requests.post(request_string)
    
    # If the code is okay we can process the data
    if (r.status_code == 200):
        # the response content is a bytes type so we need to turn it into a string
        csv_content = r.content.decode("utf-8")

        # feed the data to pandas
        the_data = StringIO(csv_content)
        stock_df = pd.read_csv(the_data)

        logging.info('Stock data found')

        stock_df.drop('Adj Close', axis=1, inplace=True)

        stock_df['Return'] = (stock_df['Close'] - stock_df['Open'])/stock_df['Open']

        for column in ['Open', 'Close', 'High', 'Low', 'Volume']:
            stock_df[f'{column}_pct'] = stock_df[column].pct_change()
            stock_df.drop(column, axis=1, inplace=True)

        stock_df['Return'] = stock_df['Return'].shift(-1)
        stock_df['Target'] = np.where(stock_df['Return'] > 0.0, 1.0, 0.0)
        stock_df.drop(['Return', 'Date'], axis=1, inplace=True)

        
        
        
        stock_df.dropna(inplace=True)


        logging.info('Stock data processed')
        return stock_df
    else:
        logging.warning('Stock data not found')
        return None
    

##################################################################
def get_start_and_end_dates(difference=5):
    ''' 
    Calculates start and end dates for our requests.
    End date is the current day while start date 
    is [difference] years ago

    Parameters:
    ------------
    difference (int, default 5) -   The difference between start date 
                                    and end date in years.

    Returns:
    -----------
    The start and end dates, both as datime objects
    '''
    logging.info("Calculating start and end dates")
    end_date = datetime.now()
    start_date = datetime.now() - relativedelta(years=difference)

    # remove convert to epoch time and remove fractional component
    end_date = round(datetime.timestamp(end_date))
    start_date = round(datetime.timestamp(start_date))
    
    logging.info(f'Start date as as {start_date} and end date as {end_date}')
    return start_date, end_date

##################################################################
def parse_command_line_arguments():
    '''
    Parses the command line arguments that come in

    Parameters:
    ------------
    None

    Returns:
    -----------
    [args], a namespace collection which contains the 
    variables [.output_directory] and [.stock_id]
    
    [args.stock_id] is a string containing the stock symbol
    that is requested

    [args.output_directory] contains a path to a directory or
    an empty string (current directory) if not provided
    by the user
    '''
    logging.info("Parsing command line arguments")

    # Make a parser and parse the arguments
    parser = argparse.ArgumentParser(description='Obtain Data From Yahoo Finance.')
    parser.add_argument('stock_id', help='The symbol of the asset to retrieve')
    parser.add_argument('--output_directory', help='The output directory for the data in .csv format')
    args = parser.parse_args()

    # If our output directory is empty, make it an empty string.
    # Otherwise, append the appropriate seperator to it
    if not (args.output_directory):
        args.output_directory = ""
    else:
        args.output_directory += os.path.sep
    
    logging.info("Command line arguments parsed")
    return args
##################################################################
def output_stock_df_to_csv(stock_df, output_directory):
    '''
    Outputs the dataframe containing the stock data
    to the specified output directory

    Parameters:
    ------------
    stock_df (pandas.core.DataFrame) -  The dataframe storing the stock
                                        data
    output_directory (string) -         The output directory where to
                                        store the data

    Returns:
    -----------
    None
    '''
    logging.info('Outputting dataframe to csv file')
    if (stock_df is not None):        
        stock_df.to_csv(output_directory, index=False)
        logging.info(f'data written to {output_directory}')
    else:
        logging.warning('data not written')
##################################################################
if __name__ == "__main__":
    # Initialize a log file
    logging.basicConfig(filename='pipeline.log', filemode="w", level=logging.DEBUG)

    # make a spark session
    #create_spark_session()

    print(sys.version)
    if not (sys.version[:3] >= "3.6"):
        logging.error("Incorrect Python version found, requires 3.6+, found "+sys.version)
    
    # Get our command line arumgnets
    args = parse_command_line_arguments()

    # Get the start and end dates
    start_date, end_date = get_start_and_end_dates()

    # Get authentication token
    auth_token = get_crumb_token()

    # If a toekn was obtained we can get the stock data and write it to file
    if (auth_token is not None):
            STOCK_ID = args.stock_id
            
            stock_df = get_stock_data(STOCK_ID, start_date, end_date, auth_token)

            output_directory = args.output_directory+STOCK_ID+".csv"
            output_stock_df_to_csv(stock_df, output_directory)
    