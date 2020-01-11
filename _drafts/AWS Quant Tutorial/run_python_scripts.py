import os

for STOCK_ID in ['AAPL', 'MSFT']:
    cmd = 'python3 /home/hadoop/obtain_data.py '+STOCK_ID+' --output_directory s3://boris-data-files'
    os.system(cmd)
