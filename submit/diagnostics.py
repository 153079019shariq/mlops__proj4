
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import timeit
import subprocess
import sys
##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 

df = pd.read_csv(os.path.join(dataset_csv_path,"finaldata.csv"))
df2 = df.loc[:,(df.columns!="exited")&(df.columns!="corporation")]


##################Function to get model predictions
def model_predictions(data_path = os.path.join(test_data_path,"testdata.csv")):
    #read the deployed model and a test dataset, calculate predictions
    prod_deployment_path = os.path.join(config['prod_deployment_path']) 
    model = pickle.load(open(os.path.join(prod_deployment_path ,'trainedmodel.pkl'), 'rb'))
    
    df = pd.read_csv(data_path)
    X_test =  df.loc[:,(df.columns!="exited")&(df.columns!="corporation")].values
    y_test = df.loc[:,"exited"].values
    pred = model.predict(X_test)
    print(pred.shape)
    return list(pred),list(y_test) #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    
    #calculate summary statistics here
    return df2.agg(['mean', 'median', 'std']) #return value should be a list containing all summary statistics

####################Function to get missing data ##############################################
def missing_data():
   nan_lis = []
   for x,y in zip(list(df2.isna().sum()),list(df2.count())):
     nan_lis.append(x/y*100)
   return nan_lis
   

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    start_timer1 = timeit.default_timer()
    os.system("python training.py")
    diff1 = timeit.default_timer() - start_timer1
    start_timer2 = timeit.default_timer()
    os.system("python ingestion.py")
    diff2 = timeit.default_timer() - start_timer2
    return [diff1,diff2] #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of
    outdated = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    return str(outdated)


     

if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data() 



    
