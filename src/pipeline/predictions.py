from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import sys, os, pickle, json
import pandas as pd
# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config


config = load_config()
model_path = config["path"]["model"]
clf = pickle.load(open(model_path, 'rb'))

algo = config['algo']
root_path = config['path']['root_path']
test_data_path = os.path.join(root_path, config['path']['data']['features'][algo]['test_data'])

test_data = pd.read_csv(test_data_path)
X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

# Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

# Generate result dictionary to evaluate model 

metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'auc': auc
}



metrics_path = config["path"]["metrics"]
create_folder_structure(metrics_path)
with open(metrics_path,'w') as file:
    json.dump(metrics,file,indent= 4)