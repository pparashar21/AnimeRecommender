from datetime import datetime 

"""
Defining logging directory and file name
"""
LOGS_DIRECTORY:str = "Logs"
LOGS_FILENAME:str = f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"

"""
Defining project level constants
"""
ARTIFACT_DIR:str = "Artifacts"
ANIME_DATASET_FILE_NAME:str = "Anime_Desctiption.csv"           # A dataset detailing anime titles and associated metadata.
RATING_DATASET_FILE_NAME:str = "Anime_Ratings.csv"              # General user ratings providing insights into viewing habits.

KAGGLE_DATASET_SLUG:str = "dbdmobile/myanimelist-dataset"
ANIME_DATASET_LINK:str = "anime-dataset-2023.csv" 
RATING_DATASET_LINK:str = "users-score-2023.csv"

"""
Defining data ingestion constants
"""
DATA_INGESTION_DIR_NAME:str = "Data_Ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "Feature_store"
DATA_INGESTION_INGESTED_DIR: str = "Ingested"

"""
Defining data cleaning constants
"""
DATA_CLEANING_DIR_NAME:str = "Data_Cleaning"
CLEANED_ANIME_FILE_NAME:str = "Cleaned_Anime_Description.csv"
MERGED_DATASET_FILE_NAME:str = "Anime_Full_House.csv"      

"""
Defining data transformation constants
"""
DATA_TRANSFORMATION_DIR_NAME:str = "Data_Transformed"
TRANSFORMED_CONTENT_MODELLING_FILE_NAME:str = "Transformed_Content_modelling.csv"

"""
Defining content model constraints
"""
MODELS_DIR_NAME = "Models"
CONTENT_MODELS_DIR_NAME:str = "Content_Models"
CONTENT_MODELS_NAME:str = "KNN_Content_model.joblib"
CONTENT_MODELS_LOOKUP_NAME:str = "KNN_Content_lookup.joblib"