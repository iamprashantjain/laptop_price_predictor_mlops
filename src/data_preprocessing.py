import os
import numpy as np
import pandas as pd
import sys
from utils import logging, CustomException


def fetch_processor(text):
    if text in ['Intel Core i7', 'Intel Core i5', 'Intel Core i3']:
        return text
    elif text.startswith('Intel'):
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'


def cat_os(inp):
    if inp in ['Windows 10', 'Windows 7', 'Windows 10 S']:
        return 'Windows'
    elif inp in ['macOS', 'Mac OS X']:
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


try:
    # Load raw train & test data
    train_data = pd.read_csv(os.path.join("data", "raw", "train_data.csv"))
    test_data = pd.read_csv(os.path.join("data", "raw", "test_data.csv"))
    logging.info("✅ Loaded raw train and test data")

    def preprocess(df):
        df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype('int32')
        df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype('float32')

        df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

        new_res = df['ScreenResolution'].str.split('x', n=1, expand=True)
        df['X_res'] = new_res[0].str.replace(',', '').str.extract(r'(\d+)').astype(int)
        df['Y_res'] = new_res[1].astype(int)

        df['ppi'] = (((df['X_res'] ** 2 + df['Y_res'] ** 2) ** 0.5) / df['Inches']).astype(float)

        df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

        df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
        df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
        df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

        df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
        df['Memory'] = df['Memory'].str.replace('GB', '')
        df['Memory'] = df['Memory'].str.replace('TB', '000')

        mem_split = df['Memory'].str.split('+', n=1, expand=True)
        df['first'] = mem_split[0].str.strip()
        df['second'] = mem_split[1].fillna('0')

        for col in ['first', 'second']:
            df[f"{col}_HDD"] = df[col].apply(lambda x: 1 if 'HDD' in x else 0)
            df[f"{col}_SSD"] = df[col].apply(lambda x: 1 if 'SSD' in x else 0)
            df[col] = df[col].str.replace(r'\D', '', regex=True).astype(int)

        df['HDD'] = df['first'] * df['first_HDD'] + df['second'] * df['second_HDD']
        df['SSD'] = df['first'] * df['first_SSD'] + df['second'] * df['second_SSD']

        df.drop(columns=[
            'Memory', 'first', 'second',
            'first_HDD', 'first_SSD',
            'second_HDD', 'second_SSD'
        ], errors='ignore', inplace=True)

        df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
        df = df[df['Gpu brand'] != 'ARM']
        df.drop(columns=['Gpu'], inplace=True)

        df['os'] = df['OpSys'].apply(cat_os)
        df.drop(columns=['OpSys'], inplace=True)

        df.reset_index(drop=True, inplace=True)
        return df

    processed_train = preprocess(train_data)
    processed_test = preprocess(test_data)

    os.makedirs("data/processed", exist_ok=True)
    processed_train.to_csv("data/processed/train_processed.csv", index=False)
    processed_test.to_csv("data/processed/test_processed.csv", index=False)
    logging.info("✅ Processed data saved to 'data/processed/'")

except Exception as e:
    logging.exception("❌ Exception occurred during data preprocessing")
    raise CustomException(e, sys)
