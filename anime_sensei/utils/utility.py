import os
import sys 
import boto3
import pandas as pd
import io
import re
import joblib
import pickle
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

def save_model_to_S3(model, key: str, bucket_name: str = "anime-recommender-system-prasoon") -> None:
    """
    Saves a serialized model object (e.g., joblib or pickle) to S3.

    Args:
        model: The trained model object to save.
        key (str): S3 key where the model will be saved.
        bucket_name (str): Name of the S3 bucket.
    """
    try:
        logging.info(f"Saving model to S3: {bucket_name}/{key}")
        s3 = boto3.client("s3")
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        s3.upload_fileobj(buffer, bucket_name, key)
        logging.info(f"Model successfully saved to S3 at: {bucket_name}/{key}")
        print(f"Model upload complete to S3 at: {bucket_name}/{key}")

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)
    
def load_model_from_S3(key: str, bucket_name: str = "anime-recommender-system-prasoon"):
    """
    Load a serialized model object (e.g., joblib or pickle) from S3.

    Args:
        key (str): Key corresponding to the model file in S3.
        bucket_name (str): Name of the S3 bucket.

    Returns:
        model: The deserialized model object.
    """
    try:
        logging.info(f"Loading model from S3: {bucket_name}/{key}")
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=key)
        model = joblib.load(io.BytesIO(response["Body"].read()))
        logging.info("Model loaded successfully from S3.")
        return model

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

def download_file_from_s3(s3_file_path: str, local_file_path: str, bucket_name: str = "anime-recommender-system-prasoon"):
    """
    Download a file from S3 to a local path.
    """
    try:
        logging.info(f"Downloading file from s3://{bucket_name}/{s3_file_path} to {local_file_path}")
        s3 = boto3.client("s3")
        
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        s3.download_file(bucket_name, s3_file_path, local_file_path)
        logging.info(f"File downloaded successfully to {local_file_path}")

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)

def load_object(file_path: str):
    """
    Load a serialized object (pickle) from a local file path.
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
            return obj
    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)
    
def save_object(file_path: str, obj: object):
    """
    Saves a Python object to a local file using pickle.
    It automatically creates the necessary directories if they don't exist.

    Args:
        file_path (str): The full local path where the object will be saved.
        obj (object): The Python object to serialize and save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object successfully saved at: {file_path}")

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)
    
def save_object_to_s3(s3_key: str, obj: object, bucket_name: str = "anime-recommender-system-prasoon"):
    """
    Serializes a Python object using pickle and saves it directly to an S3 bucket.

    Args:
        s3_key (str): The key (path) for the object in the S3 bucket.
        obj (object): The Python object to serialize and save.
        bucket_name (str): The name of the S3 bucket.
    """
    try:
        logging.info(f"Saving object to S3: s3://{bucket_name}/{s3_key}")
        # Serialize the object into a byte stream in memory
        pickle_byte_obj = pickle.dumps(obj)
        
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=pickle_byte_obj
        )
        logging.info(f"Object successfully saved to S3.")

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)

def load_object_from_s3(s3_key: str, bucket_name: str = "anime-recommender-system-prasoon") -> object:
    """
    Loads a serialized object (pickle) directly from S3.

    Args:
        s3_key (str): The key (path) of the object in the S3 bucket.
        bucket_name (str): The name of the S3 bucket.
        
    Returns:
        object: The deserialized Python object.
    """
    try:
        logging.info(f"Loading object from S3: s3://{bucket_name}/{s3_key}")
        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket_name, Key=s3_key)
        
        # Read the byte stream from the S3 object's body
        byte_stream = response['Body'].read()
        
        # Deserialize the byte stream into a Python object
        obj = pickle.loads(byte_stream)
        
        logging.info("Object loaded successfully from S3.")
        return obj

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)