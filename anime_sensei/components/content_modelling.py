import sys 
import joblib
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.utils.utility import read_file_from_S3, load_model_from_S3
from anime_sensei.constant import *
from anime_sensei.model_trainer.content_modelling_trainer import ContentModellingTrainer
from anime_sensei.entity.config_entity import ContentModellingConfig
from anime_sensei.entity.artifact_entity import DataTransformationArtifact, ContentModellingArtifact

class ContentModelling:
    def __init__(self, dta: DataTransformationArtifact, cmc: ContentModellingConfig, retrain: bool):
        try:
            self.data_transformation_artifacts = dta
            self.content_modelling_config = cmc
            self.retrain = retrain
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)

    def model_trainer(self):
        try:
            trainer = ContentModellingTrainer(
                retrain=self.retrain,
                transformed_data_path=self.data_transformation_artifacts.transformed_content_data,
                model_output_path=self.content_modelling_config.knn_model_path,
                model_lookup_path=self.content_modelling_config.knn_lookup_path,
            )
            trainer.train_and_save_knn()
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)

    def model_inference(self, anime_id: int, top_k: int = 10):
        try:
            # Load composite model artifact from S3
            logging.info("Loading content_knn artifact from S3.")
            artifact = load_model_from_S3(self.content_modelling_config.knn_model_path)

            knn_model = artifact["knn"]
            features = artifact["features"].values
            lookup = artifact["lookup"]

            # Locate index of the anime_id
            row_idx = lookup.index[lookup["anime_id"] == anime_id]
            if row_idx.empty:
                logging.error(f"Anime ID {anime_id} not found in lookup table.")
                raise KeyError(f"Anime ID {anime_id} not found in lookup table.")
            i = row_idx[0]

            # Query nearest neighbors
            dist, idxs = knn_model.kneighbors(features[i].reshape(1, -1), n_neighbors=top_k + 1)
            idxs, dist = idxs[0][1:], dist[0][1:]  

            # Format recommendations
            recs = lookup.iloc[idxs].copy()
            recs["similarity"] = 1 - dist  

            print(f"\nOriginal Anime: {lookup.loc[i, 'Name']}")
            logging.info(f"\nOriginal Anime: {lookup.loc[i, 'Name']}")
            print(f"\nTop-{top_k} Recommendations:")
            logging.info(f"\nTop-{top_k} Recommendations:")
            print(recs[["anime_id", "Name", "similarity"]])
            logging.info(recs[["anime_id", "Name", "similarity"]])

            return recs

        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)

        

if __name__ == "__main__":
    temp_transformed_path = "Artifacts/Data_Transformed/06-16-2025_22-18-15/Transformed_Content_modelling.csv"
    dta = DataTransformationArtifact(transformed_content_data=temp_transformed_path)
    cmc = ContentModellingConfig()
    demo = ContentModelling(dta, cmc, retrain=False)
    demo.model_trainer()
    demo.model_inference(anime_id=21, top_k=10)
