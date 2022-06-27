

import os
import json
import training
import scoring
import deployment
import diagnostics
import reporting
import pickle
import sys
import ingestion
import scoring
import training
##################Check and read new data
#first, read ingestedfiles.txt
with open('config.json','r') as f:
  config = json.load(f) 


prod_deploy_path   = config["prod_deployment_path"]
lis_ingest_files = []
with open( os.path.join(prod_deploy_path,"ingestedfiles.txt")) as f:
  for line in f :
    lis_ingest_files.append(line.rstrip('\n')) 



#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
flag = 0
ip_folder_path = config["input_folder_path"]
for i in os.listdir(ip_folder_path):
  if(".csv" in i ) :
     if(i not in lis_ingest_files):
       print("Source data folder are not in ingestedfiles.txt")
       flag = 1
       break
       

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if(flag==0):
  print("No new data found.Hence exiting")
  sys.exit()
else:
  ingestion.merge_multiple_dataframe()  

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latest_saved_score = 0 
with open(os.path.join(prod_deploy_path,"latestscore.txt")) as f :
  for line in f:
    latest_saved_score = float(line)
print(f'Latest_saved_score in {prod_deploy_path} is {latest_saved_score}')


print("",os.listdir(ip_folder_path))
output_folder_path = config['output_folder_path']

new_score = scoring.score_model(model_path=prod_deploy_path,test_data_path=output_folder_path,csv_file="finaldata.csv")
print(f"New_score is {new_score}")

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if(new_score>=latest_saved_score):
   print("New score is better than saved score")
   sys.exit()
else:
   training.train_model(dataset_csv_path = output_folder_path,model_path=config['output_model_path'])

  


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system("python deployment.py")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python diagnostics.py")
os.system("python reporting.py")
os.system("python apicalls.py")







