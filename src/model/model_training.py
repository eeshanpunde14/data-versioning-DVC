import pandas as pd 
import numpy as np
import os 

import pickle 
import yaml

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/processed/train_processed.csv")

# X_train = train_data.iloc[:, 0:-1].values 
# y_train = train_data.iloc[:, -1].values 

n_est = yaml.safe_load(open("params.yaml", "r"))["model_training"]["n_estimators"]

X_train = train_data.drop(columns = ['Potability'], axis = 1)
y_train = train_data['Potability']

clf = RandomForestClassifier(n_estimators = n_est)
clf.fit(X_train, y_train)

pickle.dump(clf, open("models/model.pkl", "wb"))