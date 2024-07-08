
import re
import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import os, yaml, sys, logging



# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import create_folder_structure, load_config, load_params, script_logger

logger = script_logger('data_preprocessing', 'DEBUG')

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()
    text = text.split()
    text=[lemmatizer.lemmatize(y) for y in text]
    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text=[y.lower() for y in text]
    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= removing_numbers(sentence)
    sentence= removing_punctuations(sentence)
    sentence= removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

def main():
    logger.info('Data Preprocessing Started')

    config = load_config()
    params = load_params()
    #Read Paths from the config file
    root_path = config['path']['root_path']
    train_data_path = os.path.join(root_path, config['path']['data']['raw']['train_data'])
    test_data_path = os.path.join(root_path, config['path']['data']['raw']['test_data'])

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data = normalize_text(train_data)
    test_data = normalize_text(test_data)

    # Store Processed Data

    processed_train_data_path = os.path.join(root_path, config['path']['data']['processed']['train_data'])
    processed_test_data_path = os.path.join(root_path, config['path']['data']['processed']['test_data'])


    create_folder_structure(processed_train_data_path)

    train_data.to_csv(processed_train_data_path)
    test_data.to_csv(processed_test_data_path)

    logger.info('Data Preprocessing Completed')

if __name__ == "__main__":
    main()








