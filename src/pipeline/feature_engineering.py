import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os, yaml, sys

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config

config = load_config()
root_path = config['path']['root_path']
train_data_path = os.path.join(root_path, config['path']['data']['processed']['train_data'])
test_data_path = os.path.join(root_path, config['path']['data']['processed']['test_data'])

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Generating features

X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Handle NaN values by replacing them with an empty string
X_train = np.where(pd.isnull(X_train), '', X_train)
X_test = np.where(pd.isnull(X_test), '', X_test)

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=60)

# Fit the vectorizer on the training data and transform it
X_train = vectorizer.fit_transform(X_train)

# Transform the test data using the same vectorizer and persist the data in the features folder
X_test = vectorizer.transform(X_test)

train_df = pd.DataFrame(X_train.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test.toarray())
test_df['label'] = y_test


# Path for saving features

algo = config['algo']
features_train_data_path = os.path.join(root_path, config['path']['data']['features'][algo]['train_data'])
features_test_data_path = os.path.join(root_path, config['path']['data']['features'][algo]['test_data'])
create_folder_structure(features_train_data_path)
create_folder_structure(features_test_data_path)
train_df.to_csv(features_train_data_path)
test_df.to_csv(features_test_data_path)

