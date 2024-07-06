import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import os, pickle
import sys



# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config

config = load_config()
algo = config['algo']
root_path = config['path']['root_path']
train_data_path = os.path.join(root_path, config['path']['data']['features'][algo]['train_data'])
test_data_path = os.path.join(root_path, config['path']['data']['features'][algo]['test_data'])

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values


# Define and train the XGBoost model
clf = GradientBoostingClassifier(n_estimators = 50)
clf.fit(X_train, y_train)


# Persisting Model 
model_path = os.path.join(root_path, config["path"]["model"])
print(model_path)
create_folder_structure(model_path)
pickle.dump(clf, open(model_path,'wb'))

