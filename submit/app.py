from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis 
#import predict_exited_from_saved_model
import diagnostics
import json
import os
import scoring

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    #data_path = request.get_json(force=True)["filepath"]
    data_path = request.json.get("filepath")
    y_pred,_ = diagnostics.model_predictions(data_path=data_path)
    return str(y_pred) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    val =  scoring.score_model()
    return str(val) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():     
    val = diagnostics.dataframe_summary()
    #check means, medians, and modes for each column
    return str(val) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnost():        
    #check timing and percent NA values
    val1 = diagnostics.execution_time()
    val2 = diagnostics.missing_data()
    val3 = diagnostics.outdated_packages_list()
    stri =f"The execution time is {val1} and percentage of missing data is {val2} \n {val3}  \n "
    return stri  #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
