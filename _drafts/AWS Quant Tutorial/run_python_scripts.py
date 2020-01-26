import os
import sys
import time



for STOCK_ID in ['AAPL', 'MSFT']:
    cmd = 'python3 /home/hadoop/obtain_data.py '+STOCK_ID+' --output_directory '+sys.argv[1]
    #cmd = 'python3 /home/hadoop/obtain_data.py '+STOCK_ID+' --output_directory s3://boris-data-files'
    os.system(cmd)
    time.sleep(20)
