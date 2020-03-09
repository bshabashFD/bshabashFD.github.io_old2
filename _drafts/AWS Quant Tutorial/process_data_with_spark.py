#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np
import argparse
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import DecisionTreeClassifier as SparkDecisionTreeClassifier
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext


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

    df_double_times = 0
    if stock_df is not None:
        for i in range(df_double_times): #15
            stock_df = pd.concat([stock_df, stock_df])
        

        y = stock_df['Target'].copy().values.reshape(-1, 1)
        X = stock_df
        X.drop('Target', axis=1, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print(y.shape)
        start_time = time.process_time()
        my_dt = DecisionTreeClassifier(min_samples_leaf=2)
        my_dt.fit(X_train,y_train)
        y_pred = my_dt.predict(X_test)
        end_time = time.process_time()

        delta_time = end_time - start_time

        print(stock_df.shape)
        print(delta_time/60.0)

        
        
        
        
        
        
        print("starting Spark'ing")
        sc = spark.sparkContext

        del stock_df
        stock_df = get_stock_data(STOCK_FILE)
        for i in range(df_double_times): #15
            stock_df = pd.concat([stock_df, stock_df])

        #https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa

        '''spark_stock_df = spark.createDataFrame(stock_df)
        spark_stock_df.printSchema()

        df_columns = spark_stock_df.columns
        df_columns.remove('Target')
        print(df_columns)
        assembler = VectorAssembler(inputCols=df_columns, outputCol="features")
        spark_stock_df = assembler.transform(spark_stock_df)
        spark_stock_df.printSchema()

        train, test = spark_stock_df.randomSplit([0.8, 0.2], seed = 1)'''
        df_columns = list(stock_df.columns)
        df_columns.remove('Target')
        spark_points = stock_df.apply(lambda x: LabeledPoint(x['Target'], x[df_columns].values), 
                                      axis=1)
        #spark_points = stock_df.apply(lambda x: LabeledPoint(x['Target'], x[df_columns].values))
        #print(spark_points.values)
        #exit()
        data = spark_points.values
        train_data, test_data, _, _ = train_test_split(data, data, test_size=0.2)



        train = sc.parallelize(train_data)

        # Have to turn test_data into vectors rather than label points (only features)
        print(test_data[0])
        print(test_data[0].features)
        print(dir(test_data[0]))
        new_test_data = []
        for l_point in test_data:
            new_test_data.append(l_point.features)
        test = sc.parallelize(new_test_data)


        start_time = time.process_time()
        print("Creating tree")
        #dt = SparkDecisionTreeClassifier(featuresCol = 'features', 
        #                                 labelCol = 'Target', 
        #                                 maxMemoryInMB=2048,
        #                                 minInstancesPerNode=2)
        print("Fitting")
        dtModel = DecisionTree.trainClassifier(train, 
                                               numClasses=2,
                                               categoricalFeaturesInfo={},
                                               minInstancesPerNode=2)

        
        #dtModel = dt.fit(train)
        print("Predicting")
        predictions = dtModel.predict(test).collect()
        #predictions = dtModel.transform(test)
        print("Showing Predictions")
        #predictions.printSchema()
        #predictions.select('rawPrediction', 'prediction', 'probability').show(10)
        print(predictions)
        end_time = time.process_time()

        delta_time = end_time - start_time

        print(delta_time/60.0) 
        print("ending Spark'ing")
    