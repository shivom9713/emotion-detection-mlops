import pandas as pd
from sklearn.model_selection import train_test_split
import os, yaml
import sys

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config


# Access the path from the configuration
config = load_config()
root_path = config['path']['root_path']
train_data_path = os.path.join(root_path, config['path']['data']['raw']['train_data'])
test_data_path = os.path.join(root_path, config['path']['data']['raw']['test_data'])
ingestion_url = config['path']['data']['ingestion']['url']


#importing data 

df = pd.read_csv(ingestion_url)
create_folder_structure(train_data_path)
create_folder_structure(test_data_path)

# delete tweet id
df.drop(columns=['tweet_id'],inplace=True)
final_df = df[df['sentiment'].isin(['happiness','sadness'])]
final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

train_data.to_csv(train_data_path)
test_data.to_csv(test_data_path)