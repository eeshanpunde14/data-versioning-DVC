import pandas as pd 
import numpy as np 
import os 
from sklearn.model_selection import train_test_split
import yaml



data = pd.read_csv("water_potability.csv")

test_s = yaml.safe_load(open("params.yaml"))["Data_collection"]["test_size"]


train_data, test_data = train_test_split (data, test_size=test_s, random_state = 42)


data_path = os.path.join("data", "raw")
print(data_path)

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index = False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index = False)