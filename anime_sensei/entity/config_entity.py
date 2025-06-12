import os
from datetime import datetime
from anime_sensei.constant import *

timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

ARTIFACTS_DIR:str = os.path.join(ARTIFACT_DIR, timestamp)

class DataIngestionConfig:
    """
    Configuration of the Data ingestion module to store features, training data, validation data and testing data 
    """
    def __init__(self):
        self.data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
        self.feature_store_anime_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, ANIME_DATASET_FILE_NAME) 
        self.feature_store_rating_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, RATING_DATASET_FILE_NAME)
        self.feature_store_user_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, USER_DATASET_FILE_NAME)
        self.anime_filepath: str = ANIME_DATASET_LINK
        self.rating_filepath: str = RATING_DATASET_LINK
        self.user_filepath: str = USER_DATASET_LINK
