import os
from datetime import datetime
from anime_sensei.constant import *

timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

class DataIngestionConfig:
    """
    Configuration of the Data ingestion module to store features, training data, validation data and testing data 
    """
    def __init__(self):
        self.data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME, timestamp)
        self.feature_store_anime_file_path: str = os.path.join(self.data_ingestion_dir, ANIME_DATASET_FILE_NAME) 
        self.feature_store_rating_file_path: str = os.path.join(self.data_ingestion_dir, RATING_DATASET_FILE_NAME)
        self.anime_filepath: str = ANIME_DATASET_LINK
        self.rating_filepath: str = RATING_DATASET_LINK

class DataTransformationConfig:
    """
    Configuration of the Data transformation module to store transformed data and preprocessing objects
    """
    def __init__(self):
        self.data_transformation_dir:str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, timestamp)
        self.merged_data:str = os.path.join(self.data_transformation_dir, MERGED_DATASET_FILE_NAME)

