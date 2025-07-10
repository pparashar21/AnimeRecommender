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

class DataCleaningConfig:
    """
     Configuration of the Data cleaning module to store cleaned data after mild preprocessing
    """
    def __init__(self):
        self.data_cleaned_dir:str = os.path.join(ARTIFACT_DIR, DATA_CLEANING_DIR_NAME, timestamp)
        self.merged_data:str = os.path.join(self.data_cleaned_dir, MERGED_DATASET_FILE_NAME)
        self.cleaned_anime_data:str = os.path.join(self.data_cleaned_dir, CLEANED_ANIME_FILE_NAME)

class DataTransformationConfig:
    """
    Configuration of the Data transformation module to store transformed data and transformation defaults
    """
    def __init__(self):
        self.data_transformation_dir:str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, timestamp)
        self.transformed_content_data:str = os.path.join(self.data_transformation_dir, TRANSFORMED_CONTENT_MODELLING_FILE_NAME)

        # Default parameters for transfomation for content based modelling
        self.text_col:str = "Synopsis"
        self.genre_col:str = "Genres"
        self.genre_sep:str = ","
        self.cat_cols:list[str] = ["Type", "Rating"]
        self.num_cols:list[str] = ["Score", "Episodes", "Popularity", "Favorites", "Members", "Duration_mins"]
        self.sbert_model_name: str = "all-MiniLM-L6-v2"
        self.sbert_batch_size: int = 64
        self.sbert_show_progress: bool = False

class ContentModellingConfig:
    """
    Configuration of the Content based modelling module to store KNN models and lookup paths stored after model training
    """
    def __init__(self):
        self.content_model_dir:str = os.path.join(ARTIFACT_DIR, MODELS_DIR_NAME, CONTENT_MODELS_DIR_NAME)
        self.knn_model_path:str = os.path.join(self.content_model_dir, CONTENT_MODELS_NAME)
        self.knn_lookup_path:str = os.path.join(self.content_model_dir, CONTENT_MODELS_LOOKUP_NAME)

class CollaborativeModellingConfig:
    """
    Configuration of the Collaborative modelling module to store _________
    """
    def __init__(self):
        self.collaborate_model_dir:str = os.path.join(ARTIFACT_DIR, MODELS_DIR_NAME, COLLABORATE_MODELS_DIR_NAME)
        self.neural_nets_model_path:str = os.path.join(self.collaborate_model_dir, COLLABORATE_NEURAL_NETS)
        self.neural_nets_user_encoding_path:str = os.path.join(self.collaborate_model_dir, COLLABORATE_NN_USER_ENCODING)
        self.neural_nets_anime_encoding_path:str = os.path.join(self.collaborate_model_dir, COLLABORATE_NN_ANIME_ENCODING)
        self.svd_model_path:str = os.path.join(self.collaborate_model_dir, COLLABORATE_SVD)