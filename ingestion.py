import pandas as pd
import numpy as np
import os
import json
from datetime import datetime




#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

if(not os.path.exists(output_folder_path)):
  os.makedirs(output_folder_path)


#############Function for data ingestion
def merge_multiple_dataframe():
  #check for datasets, compile them together, and write to an output file
  df_total = pd.DataFrame()
  with open(os.path.join(output_folder_path,'ingestedfiles.txt'), 'w') as f:
    for files in os.listdir(input_folder_path):
      if(".csv" in files):
        df= pd.read_csv(os.path.join(input_folder_path,files))
        df_total = pd.concat([df_total,df],ignore_index=True)
        f.write(files)
        f.write('\n')

    df_total = df_total.drop_duplicates()
    df_total.to_csv(os.path.join(output_folder_path,"finaldata.csv"),index=False,header=True)

if __name__ == '__main__':
    merge_multiple_dataframe()
