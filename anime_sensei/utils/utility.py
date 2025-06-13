import os
import sys 
import pandas as pd
import re
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.constant import * 
import kagglehub
from kagglehub import KaggleDatasetAdapter

def get_data_from_kaggle(file_name:str) -> pd.DataFrame:
    """
    Downloading a kaggle dataset as a Pandas DataFrame
    
    Args:
        file_name (str): The file name to be downloaded from Kaggle inside the mentioned KAGGLE_DATASET_SLUG

    Returns:
        df (pd.DataFrame) : The downloaded DataFrame from Kaggle
    """
    try:
        logging.info(f"Extracting {file_name} from Kaggle")
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            KAGGLE_DATASET_SLUG,
            file_name,
        )
        logging.info(f"Shape of the dataframe: {df.shape}")
        logging.info(f"Column names: {df.columns}")
        logging.info(f"Preview of the DataFrame:\n{df.head()}")
        logging.info("Data fetched successfully from Kaggle.")
        
        return df
    except Exception as e:
        logging.error(ExceptionHandler(e,sys))
        raise ExceptionHandler(e, sys)

def export_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Saves a given Pandas DataFrame to a CSV file.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame should be stored.
    """
    try:
        logging.info(f"Saving DataFrame to file: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        dataframe.to_csv(file_path, index=False, header=True)
        logging.info(f"DataFrame saved successfully to {file_path}.")
        return dataframe
    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)
    

def parse_duration_to_minutes(duration):
    """
    Function to conver the categorical column 'Duration' to a numeric column by extracting specific minutes per episode/movie

    Args:
        duration (str): The name of the dataframe on which the function is to be applied

    Returns: 
        minutes_per_show (int) : Returns the minutes per episode/minutes extracted based on given categorical information
    """
    duration = duration.lower()
    hours = 0
    minutes = 0

    # Match hours (e.g., '2 hr' or '2hrs')
    hr_match = re.search(r'(\d+)\s*hr', duration)
    if hr_match:
        hours = int(hr_match.group(1))

    # Match minutes (e.g., '33 min')
    min_match = re.search(r'(\d+)\s*min', duration)
    if min_match:
        minutes = int(min_match.group(1))

    minutes_per_show = hours * 60 + minutes
    return minutes_per_show