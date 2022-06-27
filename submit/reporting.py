import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics
from sklearn.metrics import ConfusionMatrixDisplay

###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
output_model_path = config["output_model_path"]



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    y_pred,y_test = diagnostics.model_predictions()
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))



if __name__ == '__main__':
    score_model()
