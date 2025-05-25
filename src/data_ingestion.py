import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import logging, CustomException
import sys
import yaml

try:
    test_size = yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']
        
    df = pd.read_csv('laptop_data.csv')
    logging.info("reading data from source")
    
    df.drop(columns=['Unnamed: 0'],inplace=True)
    logging.info("dropping un-necessary col")
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    logging.info("splitting train test")

    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    train_path = os.path.join("data", "raw", "train_data.csv")
    test_path = os.path.join("data", "raw", "test_data.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    logging.info("Saved train and test data to 'data/raw/'")
    
except Exception as e:
    logging.info("Exception occurred during evaluation script execution")
    raise CustomException(e, sys)