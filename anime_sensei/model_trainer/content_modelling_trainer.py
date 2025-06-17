import sys 
from sklearn.neighbors import NearestNeighbors
import joblib
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.utils.utility import read_file_from_S3, save_model_to_S3
from anime_sensei.constant import *

class ContentModellingTrainer:
    def __init__(self, retrain: bool, transformed_data_path: str, model_output_path: str, model_lookup_path: str):
        self.retrain = retrain
        self.transformed_data_path = transformed_data_path
        self.model_output_path = model_output_path
        self.model_lookup_path = model_lookup_path

    def train_and_save_knn(self):
        if not self.retrain:
            logging.info("Retrain is set to False. Skipping KNN training.")
            return

        try:
            # Load feature data
            logging.info("Reading transformed data from S3.")
            df = read_file_from_S3(self.transformed_data_path)
            feats = df.drop(columns=["anime_id", "Name"], errors="ignore")
            feat_values = feats.values

            # Prepare lookup
            logging.info("Preparing lookup table.")
            lookup = df[["anime_id", "Name"]].reset_index(drop=True)
            assert len(feat_values) == len(lookup), "Feature matrix and lookup index mismatch."

            # Train KNN model
            logging.info("Training NearestNeighbors model.")
            knn = NearestNeighbors(n_neighbors=21, metric="cosine", algorithm="brute")
            knn.fit(feat_values)

            # Combine all artifacts
            artifact = {
                "knn": knn,
                "features": feats,
                "lookup": lookup
            }

            # Upload single artifact to S3
            logging.info("Uploading combined model artifact to S3.")
            save_model_to_S3(artifact, self.model_output_path)

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)