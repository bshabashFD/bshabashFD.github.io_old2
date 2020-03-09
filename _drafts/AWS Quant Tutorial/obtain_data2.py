#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import time
import yfinance as yf # https://github.com/ranaroussi/yfinance


import logging



##################################################################
def get_stock_data(STOCK_ID, start_time, end_time):
    ''' 
    Gets the stock data for the request stock within the range
    specified by [start_time] and [end_time]

    Parameters:
    ------------
    STOCK_ID (string) - The asset for which data is requested
                        (e.g. AAPL, MSFT)
    start_time (int) -  The start time as an epoch time (e.g. 1420229214)
    end_time (int) -    The end time as an epoch time

    Returns:
    -----------
    a pandas.core.DataFrame of the stock historic
    data if successful, None otherwise
    '''
    logging.info(f'Obtaining data for symbol {STOCK_ID}')

    # Grab the data
    stock_df = yf.download(STOCK_ID, 
                            start=start_time, 
                            end=end_time, 
                            progress=False)

    logging.info('Stock data found')

    stock_df.drop('Adj Close', axis=1, inplace=True)

    stock_df['Return'] = (stock_df['Close'] - stock_df['Open'])/stock_df['Open']

    for column in ['Open', 'Close', 'High', 'Low', 'Volume']:
        stock_df[f'{column}_pct'] = stock_df[column].pct_change()
        stock_df.drop(column, axis=1, inplace=True)

    stock_df['Return'] = stock_df['Return'].shift(-1)
    stock_df['Target'] = np.where(stock_df['Return'] > 0.0, 1.0, 0.0)
    stock_df.reset_index(inplace=True)
    stock_df.drop(['Return', 'Date'], axis=1, inplace=True)

        
        
        
    stock_df.dropna(inplace=True)


    logging.info('Stock data processed')
    return stock_df
    

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

    start_date = f'{start_date.year}-{start_date.month}-{start_date.day}'
    end_date = f'{end_date.year}-{end_date.month}-{end_date.day}'
    
    
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



    print(sys.version)
    if not (sys.version[:3] >= "3.6"):
        logging.error("Incorrect Python version found, requires 3.6+, found "+sys.version)
    
    # Get our command line arumgnets
    args = parse_command_line_arguments()

    # Get the start and end dates
    start_date, end_date = get_start_and_end_dates()


    STOCK_ID = args.stock_id
    
    stock_df = get_stock_data(STOCK_ID, start_date, end_date)

    output_directory = args.output_directory+STOCK_ID+".csv"
    output_stock_df_to_csv(stock_df, output_directory)
    