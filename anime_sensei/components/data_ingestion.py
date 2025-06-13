import sys
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.entity.config_entity import DataIngestionConfig
from anime_sensei.entity.artifact_entity import DataIngestionArtifact
from anime_sensei.utils.utility import export_dataframe_to_csv, get_data_from_kaggle

class DataIngestion:
    """
    A class responsible for data ingestion in the anime recommender system.

    This class fetches data from KaggleHub datasets, and exports the processed data to storage for further use in the pipeline.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration settings for data ingestion. 
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)

    def ingest_data(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process, fetching datasets and saving them to the feature store. 
        Returns:
            DataIngestionArtifact: An artifact containing paths to the ingested datasets. 
        """
        try:
            # Load anime and rating data from Hugging Face datasets
            anime_df = get_data_from_kaggle(self.data_ingestion_config.anime_filepath)
            rating_df = get_data_from_kaggle(self.data_ingestion_config.rating_filepath)
            user_df = get_data_from_kaggle(self.data_ingestion_config.user_filepath)

            # Export data to DataFrame
            export_dataframe_to_csv(anime_df, file_path=self.data_ingestion_config.feature_store_anime_file_path)
            export_dataframe_to_csv(rating_df, file_path=self.data_ingestion_config.feature_store_rating_file_path)
            export_dataframe_to_csv(user_df, file_path=self.data_ingestion_config.feature_store_user_file_path)

            # Create artifact to store data ingestion info
            dataingestionartifact = DataIngestionArtifact(
                feature_store_anime_file_path=self.data_ingestion_config.feature_store_anime_file_path,
                feature_store_rating_file_path=self.data_ingestion_config.feature_store_rating_file_path,
                feature_store_user_file_path=self.data_ingestion_config.feature_store_user_file_path
            ) 
            return dataingestionartifact

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)
        
if __name__ == "__main__":
    # Testing flow of data ingestion
    d1 = DataIngestionConfig()
    demo = DataIngestion(d1)
    sol = demo.ingest_data()

    print(sol.feature_store_anime_file_path)
    print(sol.feature_store_rating_file_path)
    print(sol.feature_store_user_file_path)