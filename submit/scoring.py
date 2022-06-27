from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])

model_path      = os.path.join(config["output_model_path"])
model_path_save = os.path.join(config["output_model_path"])


#################Function for model scoring
def score_model(model_path=model_path,test_data_path=test_data_path,csv_file="testdata.csv"):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    model = pickle.load(open(os.path.join(model_path ,'trainedmodel.pkl'), 'rb'))

    df = pd.read_csv(os.path.join(test_data_path,csv_file))
    X_test =  df.loc[:,(df.columns!="exited")&(df.columns!="corporation")].values
    y_test =  df["exited"].values
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
   
    with open(os.path.join(model_path_save,"latestscore.txt"), 'w') as f:
      f.write(str(f1))
    return f1

if __name__=="__main__":
  score_model(model_path,test_data_path)
