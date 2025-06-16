import sys 
import numpy as np
import pandas as pd 
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.utils.utility import export_dataframe_to_csv, parse_duration_to_minutes, read_file_from_S3, save_data_to_S3
from anime_sensei.constant import *
from anime_sensei.entity.config_entity import DataTransformationConfig
from anime_sensei.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact

class DataTransformation:
    """
    Class for performing data transformation and its various intermediate steps
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transform_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transform_config = data_transform_config
        except Exception as e:
            logging.error(ExceptionHandler(e))
            raise ExceptionHandler(e)

    @staticmethod
    def clean_anime_data(anime:pd.DataFrame) -> pd.DataFrame:
        """
        Function to perform data cleaning and mild feature engineering for the anime dataset

        Args:
            anime (pd.DataFrame) : The original anime dataset
        
        Returns:
            anime (pd.DataFrame) : Cleaned and transformed anime dataset
        """
        try:
            logging.info(f"Cleaning ANIME dataset, initial shape of dataset: {anime.shape}")

            anime.replace("UNKNOWN", np.nan, inplace = True)
            anime['Synopsis'].replace("No description available for this anime.", np.nan, inplace = True)
            anime['Scored By'].replace(np.nan, 0, inplace = True)

            anime.drop(['English name', 'Other name', 'Premiered', 'Producers', 'Licensors', 'Studios', 'Source', 'Aired', 'Status', 'Rank'], axis = 1, inplace=True)
            anime.dropna(subset=['Synopsis'], inplace=True)

            average_rating = anime['Score'][anime['Score']!=np.nan]
            average_rating = average_rating.astype('float')
            mean = round(average_rating.mean(), 2)
            anime['Score'].replace(np.nan, mean, inplace = True)
            anime['Score'] = anime['Score'].astype('float64')

            anime['Episodes'].replace(np.nan, 0.0, inplace = True)
            anime['Episodes'] = anime['Episodes'].astype('float64')

            anime['Type'].replace(np.nan, "UNKNOWN", inplace = True)
            anime['Genres'].replace(np.nan, "UNKNOWN", inplace = True)

            mode_ratings = anime['Rating'].value_counts().idxmax()
            anime['Rating'].replace(np.nan, mode_ratings, inplace = True)

            anime['Duration_mins'] = anime['Duration'].apply(parse_duration_to_minutes)
            anime.drop('Duration', axis = 1, inplace = True)

            logging.info(f"Cleaning done! Shape of resultant dataset: {anime.shape}")
            return anime
        
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)
        
    @staticmethod
    def merge_data(anime:pd.DataFrame, ratings:pd.DataFrame)->pd.DataFrame:
        """
        Function to merge the anime and the ratings dataframe to get a full view to train for the recommender system

        Args:
            anime (pd.DataFrame) : Cleaned and transformed anime dataset
            ratings (pd.DataFrame) : Cleaned and transformed ratings dataset
        
        Returns:
            merged (pd.DataFrame) : Merged dataframe 
        """
        try:
            logging.info(f"Started merging the datasets.\nShape of ANIME dataset: {anime.shape} \nShape of RATINGS dataset: {ratings.shape}")
            merged = pd.merge(anime, ratings, on='anime_id', how='inner', indicator=True)
            merged.drop(['Anime Title'], axis=1, inplace = True)
            logging.info(f"Merged successful! Shape of resultant dataframe: {merged.shape}")
            logging.info(f"Columns of the resultant datafram: {merged.columns}")
            return merged

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)
        
    def initiate_transformation(self)->DataTransformationArtifact:
        """
        Runner function for the data transformation module
        
        Returns:
            DataTransformationArtifact : The artifact containing paths to the transformed dataset
        """
        try:
            anime_df = read_file_from_S3(self.data_ingestion_artifact.feature_store_anime_file_path)
            ratings_df = read_file_from_S3(self.data_ingestion_artifact.feature_store_rating_file_path)
            
            anime_df = DataTransformation.clean_anime_data(anime_df)
            save_data_to_S3(anime_df, self.data_transform_config.cleaned_anime_data)

            merged_df = DataTransformation.merge_data(anime_df, ratings_df)
            save_data_to_S3(merged_df, self.data_transform_config.merged_data)
            data_transformation_artifact = DataTransformationArtifact(
                merged_data=self.data_transform_config.merged_data,
                cleaned_anime_data=self.data_transform_config.cleaned_anime_data
            )

            return data_transformation_artifact
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)
        

if __name__ == "__main__":
    # In the pipeline, these temp_paths would be accessed from DataIngestionArtifacts class that would have been created
    # at the data ingestion module, adding these here to test out functionality
    temp_anime_path = 'Artifacts/Data_ingestion/06-14-2025_22-09-04/Anime_Desctiption.csv'
    temp_rating_path = 'Artifacts/Data_ingestion/06-14-2025_22-09-04/Anime_Ratings.csv'
    data_ingest = DataIngestionArtifact(temp_anime_path, temp_rating_path)
    data_transform = DataTransformationConfig()
    demo = DataTransformation(data_ingest, data_transform)
    artifacts = demo.initiate_transformation()

    print(f"Cleaned anime file saved at : {artifacts.cleaned_anime_data}")
    print(f"Merged file saved at : {artifacts.merged_data}")