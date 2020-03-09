#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import argparse
import time

from sklearn.model_selection import train_test_split


from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark import SparkContext


import logging




def create_spark_session():
    '''
    taken from https://towardsdatascience.com/production-data-processing-with-apache-spark-96a58dfd3fe7
    Create spark session.
        
    Returns:
        spark (SparkSession) - spark session connected to AWS EMR cluster
    '''
    spark = SparkSession.builder.appName('BorisApp').getOrCreate()
    
    return spark

##################################################################
##################################################################
##################################################################
def get_stock_data(STOCK_FILE):
    ''' 
    Gets the stock data for the request stock 

    Parameters:
    ------------
    STOCK_FILE (string) - The asset for which data is requested
                        (e.g. AAPL.csv, MSFT.csv)

    Returns:
    -----------
    a pandas.core.DataFrame of the stock historic
    data if successful, None otherwise
    '''
    logging.info(f'Obtaining data for symbol {STOCK_FILE}')

    # Grab the data
    try:
        stock_df = pd.read_csv(STOCK_FILE)

        logging.info('Stock data found')

        return stock_df
    except FileNotFoundError:
        logging.info('Stock data not found')

        return None
    

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

    '''
    logging.info("Parsing command line arguments")

    # Make a parser and parse the arguments
    parser = argparse.ArgumentParser(description='Process data from a csv file.')
    parser.add_argument('stock_file', help='The path to the asset csv file')
    args = parser.parse_args()

    
    logging.info("Command line arguments parsed")
    return args
##################################################################
##################################################################
if __name__ == "__main__":
    # Initialize a log file
    logging.basicConfig(filename='pipeline.log', filemode="w", level=logging.DEBUG)

    # make a spark session
    spark = create_spark_session()
    

    if not (sys.version[:3] >= "3.6"):
        logging.error("Incorrect Python version found, requires 3.6+, found "+sys.version)
    
    # Get our command line arumgnets
    args = parse_command_line_arguments()


    # We can get the stock data and write it to file

    STOCK_FILE = args.stock_file
    
    stock_df = get_stock_data(STOCK_FILE)

    df_double_times = 5
    if stock_df is not None:
        
        print("starting Spark'ing")

        stock_df = get_stock_data(STOCK_FILE)

        #https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
        print('creating spark df')
        spark_stock_df = spark.createDataFrame(stock_df)

        print('doubling')
        for i in range(df_double_times): #15
            print(f'doubling {i+1}')
            spark_stock_df = spark_stock_df.union(spark_stock_df)
        print('doubled')

        spark_stock_df.printSchema()
        

        df_columns = spark_stock_df.columns
        df_columns.remove('Target')
        print(df_columns)
        assembler = VectorAssembler(inputCols=df_columns, outputCol="features")
        spark_stock_df = assembler.transform(spark_stock_df)
        spark_stock_df.printSchema()

        train, test = spark_stock_df.randomSplit([0.8, 0.2], seed = 1)

        train = train.repartition(10)
        print(train.rdd.getNumPartitions())
        exit()

        start_time = time.time()
        print("Creating tree")
        dt = DecisionTreeClassifier(featuresCol = 'features', 
                                    labelCol = 'Target', 
                                    maxDepth=30,
                                    maxMemoryInMB=2048,
                                    cacheNodeIds=True,
                                    minInstancesPerNode=2)
        print("Fitting")

        
        dtModel = dt.fit(train)
        print("Predicting")
        predictions = dtModel.transform(test)
        
        end_time = time.time()
        print("Showing Predictions")
        predictions.printSchema()
        predictions.select('rawPrediction', 'prediction', 'probability').show(20)

        delta_time = end_time - start_time

        print(f'run-time: {delta_time/60.0}') 
        print("ending Spark'ing")
    