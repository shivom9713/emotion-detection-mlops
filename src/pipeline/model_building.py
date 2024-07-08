import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import os, pickle
import sys



# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config, load_params, script_logger

logger = script_logger('Model building', 'DEBUG')


def train_model(train_data, test_data, n_estimators = 50, learning_rate = 0.2 ):
    X_train = train_data.iloc[:,0:-1].values
    y_train = train_data.iloc[:,-1].values

    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values

    # Define and train the XGBoost model
    clf = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)
    logger.info(f'Model trained successfully')
    return clf

def save_model(clf, model_path):
    create_folder_structure(model_path)
    pickle.dump(clf, open(model_path,'wb'))

    logger.info(f'Model saved at {model_path}')


def main():

    config = load_config()
    params = load_params()
    n_estimators = params["model_building"]["n_estimators"]
    learning_rate =  params["model_building"]["learning_rate"]
    root_path = config['path']['root_path']
    train_data_path = os.path.join(root_path, config['path']['data']['features']['train_data'])
    test_data_path = os.path.join(root_path, config['path']['data']['features']['test_data'])

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    clf = train_model(train_data, test_data, n_estimators, learning_rate)

    # Persisting Model 
    model_path = os.path.join(root_path, config["path"]["model"])
    print(model_path)

    save_model(clf, model_path)



if __name__ == "__main__":
    main()







