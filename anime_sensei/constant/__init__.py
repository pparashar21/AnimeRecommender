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
USER_DATASET_FILE_NAME:str = "Anime_Users.csv"                  # Dataset about the users and their information
MERGED_DATASET_FILE_NAME:str = "Anime_Full_House.csv"      

KAGGLE_DATASET_SLUG:str = "dbdmobile/myanimelist-dataset"
ANIME_DATASET_LINK:str = "anime-dataset-2023.csv" 
RATING_DATASET_LINK:str = "users-score-2023.csv"
USER_DATASET_LINK:str = "users-details-2023.csv"

"""
Defining data ingestion constants
"""
DATA_INGESTION_DIR_NAME:str = "Data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "Feature_store"
DATA_INGESTION_INGESTED_DIR: str = "Ingested"

"""
Defining data transformation constants
"""