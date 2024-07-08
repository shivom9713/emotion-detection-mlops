import pandas as pd
from sklearn.model_selection import train_test_split 
import sys, logging, os



# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config, load_params, script_logger

logger = script_logger('Data Ingestion', 'DEBUG')


def read_data(ingestion_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(ingestion_url)
        logger.info(f"data read from ingestion source")
        return df
    except Exception as e:
        logger.error(f"Error reading data from {ingestion_url}: {e}")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        return final_df
    except KeyError as e:  #Incase KeyError Occurs
        logger.error(f"Column error: {e}")
        raise
    except Exception as e:  #Incase General Error Occurs
        logger.error(f"Error processing data: {e}")
        raise


def main() -> None:
    try:
        # Access the path from the configuration
        config = load_config()
        params = load_params()
        test_size = params["data_ingestion"]["test_size"]

        root_path = config['path']['root_path']
        train_data_path = os.path.join(root_path, config['path']['data']['raw']['train_data'])
        test_data_path = os.path.join(root_path, config['path']['data']['raw']['test_data'])
        ingestion_url = config['path']['data']['ingestion']['url']

        df = read_data(ingestion_url)
        final_df = process_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        create_folder_structure(train_data_path)
        create_folder_structure(test_data_path)

        train_data.to_csv(train_data_path)
        test_data.to_csv(test_data_path)
        logger.info(f"Raw Data Persisted at: {train_data_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except KeyError as e:
        print(f"Key error in configuration: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
