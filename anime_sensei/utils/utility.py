import os
import sys 
import boto3
import pandas as pd
import io
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

def read_file_from_S3(key:str, bucket_name:str = "anime-recommender-system-prasoon")->pd.DataFrame:
    """
        Reads a pandas dataframe from the mentioned S3 location

        Args:
            key (str) : Key corresponding to the S3 location
            bucket_name(str) : Bucket name of the S3 location
        
        Returns:
            df (pd.DataFrame) : Returned pandas dataframe from S3 
    """
    try: 
            logging.info(f"Reading data from : {bucket_name}/{key}")
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=bucket_name, Key=key)
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            logging.info(f"Data loaded successfully from S3!")
            return df

    except Exception as e:
        logging.error(ExceptionHandler(e,sys))
        raise ExceptionHandler(e,sys)

def save_data_to_S3(dataframe: pd.DataFrame, key: str, bucket_name:str = "anime-recommender-system-prasoon") -> None:
    """
    Saves a given Pandas DataFrame to the mentioned S3 location
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame should be stored.
        bucket_name (str): Specify the name of the AWS bucket, a default bucket has been set
    """
    try:
        logging.info(f"Saving DataFrame to file: {bucket_name, key}")
        s3 = boto3.client('s3')
        csv_buffer = io.BytesIO()
        dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        s3.upload_fileobj(csv_buffer, bucket_name, key)
        logging.info(f"DataFrame saved successfully to S3 at: {bucket_name}/{key}")
        print(f"Upload Successfully to S3 at location : {bucket_name}/{key}")  

    except Exception as e:
        logging.error(ExceptionHandler(e,sys))
        raise ExceptionHandler(e,sys) 

def export_dataframe_to_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    Saves a given Pandas DataFrame to a local CSV file.
    
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