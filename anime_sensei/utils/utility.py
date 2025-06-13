import os
import sys 
import pandas as pd
import joblib
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
    df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    KAGGLE_DATASET_SLUG,
    file_name,
    )
    logging.info(f"Extracting {file_name} from Kaggle")
    logging.info(f"Shape of the dataframe: {df.shape}")
    logging.info(f"Column names: {df.columns}")
    logging.info(f"Preview of the DataFrame:\n{df.head()}")
    logging.info("Data fetched successfully from Kaggle.")
    return df

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
        logging.error(f"Error saving DataFrame to {file_path}: {e}")
        raise ExceptionHandler(e, sys)