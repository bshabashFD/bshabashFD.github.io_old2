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
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


import logging




def create_spark_session():
    '''
    taken from https://towardsdatascience.com/production-data-processing-with-apache-spark-96a58dfd3fe7
    Create spark session.
        
    Returns:
        spark (SparkSession) - spark session connected to AWS EMR cluster
    '''
    spark = SparkSession.builder.config("spark.jars.packages", 
                                        "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()
    return spark

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
##################################################################
def get_stock_data(STOCK_URL):
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
    
    stock_df = pd.read_csv(STOCK_URL)

    logging.info('Stock data found')

    return stock_df
    

##################################################################

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
    create_spark_session()

    print(sys.version)
    if not (sys.version[:3] >= "3.6"):
        logging.error("Incorrect Python version found, requires 3.6+, found "+sys.version)
    
    # Get our command line arumgnets
    args = parse_command_line_arguments()


    # If a toekn was obtained we can get the stock data and write it to file
    if (auth_token is not None):
            STOCK_ID = args.stock_id
            
            stock_df = get_stock_data(STOCK_ID, start_date, end_date, auth_token)

            output_directory = args.output_directory+STOCK_ID+".csv"
            output_stock_df_to_csv(stock_df, output_directory)
    