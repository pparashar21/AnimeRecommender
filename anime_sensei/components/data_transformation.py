import sys 
import numpy as np
import pandas as pd 
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.utils.utility import read_file_from_S3, save_data_to_S3
from anime_sensei.constant import *
from anime_sensei.entity.config_entity import DataTransformationConfig
from anime_sensei.entity.artifact_entity import DataCleaningArtifact, DataTransformationArtifact

class DataTransformation:
    """
    Class for performing data transformation and its various intermediate steps
    """
    def __init__(self, data_cleaning_artifact: DataCleaningArtifact, data_transform_config: DataTransformationConfig):
        try:
            self.data_cleaning_artifact = data_cleaning_artifact
            self.data_transform_config = data_transform_config
            self.genre_categories = []
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)

    @staticmethod
    def transform_text_col(self, df: pd.DataFrame) -> np.ndarray:
        try:
            cleaned_text = df[self.data_transform_config.text_col].fillna("").str.replace(
                re.compile(r"<.*?>|\\n"), " ", regex=True
            )
            model = SentenceTransformer(self.data_transform_config.sbert_model_name)
            return model.encode(
                cleaned_text.tolist(),
                batch_size=self.data_transform_config.sbert_batch_size,
                show_progress_bar=self.data_transform_config.sbert_show_progress,
                normalize_embeddings=True
            )
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)

    @staticmethod
    def transform_genre_col(self, df: pd.DataFrame) -> np.ndarray:
        try:
            genres = (
                df[self.data_transform_config.genre_col]
                .fillna("")
                .str.split(self.data_transform_config.genre_sep)
                .apply(lambda lst: {g.strip() for g in lst})
            )
            genre_set = (
                df[self.data_transform_config.genre_col]
                .fillna("")
                .str.split(self.data_transform_config.genre_sep)
                .explode()
                .str.strip()
                .loc[lambda s: s != ""]
            )
            self.genre_categories = sorted(genre_set.unique())
            genre_matrix = np.zeros((len(df), len(self.genre_categories)), dtype=np.float32)
            for row, genre_vals in enumerate(genres):
                for col, genre in enumerate(self.genre_categories):
                    if genre in genre_vals:
                        genre_matrix[row, col] = 1.0
            return genre_matrix
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)

    @staticmethod
    def transform_cat_cols(self, df: pd.DataFrame) -> np.ndarray:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            return encoder.fit_transform(df[self.data_transform_config.cat_cols])
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)

    @staticmethod
    def transform_num_cols(self, df: pd.DataFrame) -> np.ndarray:
        try:
            scaler = MinMaxScaler()
            return scaler.fit_transform(df[self.data_transform_config.num_cols])
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)

    def data_transformation_content_modelling(self):
        try:
            df = read_file_from_S3(self.data_cleaning_artifact.cleaned_anime_data)
            text_features = self.transform_text_col(self, df)
            logging.info("Feature Engineering - Content Model : Text features extracted")
            genre_features = self.transform_genre_col(self, df)
            logging.info("Feature Engineering - Content Model : Genre features extracted")
            cat_features = self.transform_cat_cols(self, df)
            logging.info("Feature Engineering - Content Model : Categotical features extracted")
            num_features = self.transform_num_cols(self, df)
            logging.info("Feature Engineering - Content Model : Numerical features extracted")

            all_features = np.hstack([text_features, genre_features, cat_features, num_features])
            feature_columns = (
                [f"sbert_{i}" for i in range(text_features.shape[1])] +
                [f"genre_{g}" for g in self.genre_categories] +
                list(pd.get_dummies(df[self.data_transform_config.cat_cols]).columns) +
                self.data_transform_config.num_cols
            )

            feat_df = pd.DataFrame(all_features, columns=feature_columns, index=df.index)
            # Prepend anime_id and Name columns from original df
            feat_df.insert(0, "Name", df["Name"].values)
            feat_df.insert(0, "anime_id", df["anime_id"].values)
            output_path = self.data_transform_config.transformed_content_data
            save_data_to_S3(feat_df, output_path)
            
            return DataTransformationArtifact(transformed_content_data=output_path)
        except Exception as e:
            logging.error(ExceptionHandler(e,sys))
            raise ExceptionHandler(e,sys)
        

if __name__ == "__main__":
    # In the pipeline, these temp_paths would be accessed from DataCleaningArtifact class that would have been created
    # at the data cleaning module, adding these here to test out functionality
    temp_cleaning_artifacts = "Artifacts/Data_Cleaning/06-16-2025_20-05-59/Cleaned_Anime_Description.csv"
    temp_merged_artifacts = 'Artifacts/Data_Cleaning/06-16-2025_20-05-59/Anime_Full_House.csv'
    dca = DataCleaningArtifact(merged_data=temp_merged_artifacts, cleaned_anime_data=temp_cleaning_artifacts)
    dtc = DataTransformationConfig()
    demo = DataTransformation(dca, dtc)
    print("Starting data transformation")
    dta = demo.data_transformation_content_modelling()

    print(f"Transformation done and data saved at Key : {dta.transformed_content_data}")