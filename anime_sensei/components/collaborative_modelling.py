import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from anime_sensei.loggers.logging import logging
from anime_sensei.exception.handler import ExceptionHandler
from anime_sensei.utils.utility import read_file_from_S3, download_file_from_s3, load_object, save_data_to_S3, load_object_from_s3
from anime_sensei.entity.config_entity import CollaborativeModellingConfig
from anime_sensei.entity.artifact_entity import DataIngestionArtifact, DataCleaningArtifact, CollaborativeModellingArtifact
from anime_sensei.model_trainer.collaborative_model_trainer import CollaborativeModelingTrainer, RecommenderNet

class CollaborativeModelling:
    """
    Main component for collaborative filtering.
    Orchestrates training (including S3 upload) and provides inference.
    """
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_cleaning_artifact: DataCleaningArtifact,
                 model_trainer_artifact: CollaborativeModellingArtifact = None):
        logging.info("Initializing CollaborativeModelling component.")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_cleaning_artifact = data_cleaning_artifact
        self.model_trainer_artifact = model_trainer_artifact
        
        self.clean_df = read_file_from_S3(self.data_cleaning_artifact.cleaned_anime_data)
        self.ratings_df = read_file_from_S3(self.data_ingestion_artifact.feature_store_rating_file_path)
        
        anime_full = pd.merge(self.clean_df, self.ratings_df, on='anime_id', suffixes=("_anime", "_user"), how='inner')
        if 'Anime Title' in anime_full.columns:
            anime_full.drop(['Anime Title'], axis=1, inplace=True)
        self.anime_full = anime_full
        logging.info("Base dataframes loaded for collaborative component.")


    def get_nn_recommendations(self, user_id: int, N: int = 10) -> pd.DataFrame:
        logging.info(f"Generating NN recommendations for user_id: {user_id}")
        if not self.model_trainer_artifact: 
            raise ValueError("Model trainer artifact not provided.")
        
        local_model_dir = "/tmp/collaborative_nn"; os.makedirs(local_model_dir, exist_ok=True)
        local_weights_path = os.path.join(local_model_dir, "nn_weights.weights.h5")

        download_file_from_s3(self.model_trainer_artifact.neural_nets_model_name, local_weights_path)

        user_encoder = load_object_from_s3(self.model_trainer_artifact.neural_nets_user_encoder)
        anime_encoder = load_object_from_s3(self.model_trainer_artifact.neural_nets_anime_encoder)

        if user_id not in user_encoder.classes_: 
            raise ValueError(f"User ID {user_id} not seen during training.")
        
        model = RecommenderNet(len(user_encoder.classes_), len(anime_encoder.classes_))
        model.load_weights(local_weights_path)

        logging.info("NN Model and encoders loaded for inference.")

        encoded_user = user_encoder.transform([user_id])[0]
        watched_anime_ids = set(self.anime_full[self.anime_full['user_id'] == user_id]['anime_id'].values)
        
        unseen_animes = self.clean_df[~self.clean_df['anime_id'].isin(watched_anime_ids)].copy()
        unseen_animes = unseen_animes[unseen_animes['anime_id'].isin(anime_encoder.classes_)]
        
        if unseen_animes.empty: 
            return pd.DataFrame()
        
        unseen_animes['anime_encoded'] = anime_encoder.transform(unseen_animes['anime_id'])
        
        user_input = np.full(len(unseen_animes), encoded_user)
        anime_input = unseen_animes['anime_encoded'].values
        
        predicted_scores = model.predict([user_input, anime_input], batch_size=512, verbose=0).flatten()
        unseen_animes['predicted_rating'] = predicted_scores
        
        return unseen_animes.sort_values(by='predicted_rating', ascending=False).head(N)[['anime_id', 'Name', 'Genres', 'Score', 'predicted_rating']]

    def get_svd_recommendations(self, user_id: int, N: int = 10) -> pd.DataFrame:
        logging.info(f"Generating SVD recommendations for user_id: {user_id}")
        if not self.model_trainer_artifact: 
            raise ValueError("Model trainer artifact not provided.")
        
        svd_model = load_object_from_s3(self.model_trainer_artifact.svd_model_name)
        logging.info("SVD model loaded directly from S3.")

        all_animes = set(self.clean_df['anime_id'].unique())
        watched = set(self.anime_full[self.anime_full['user_id'] == user_id]['anime_id'].values)
        unseen = list(all_animes - watched)
        predictions = [svd_model.predict(user_id, aid) for aid in unseen]
        top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:N]
        top_anime_ids = [pred.iid for pred in top_preds]
        top_df = self.clean_df[self.clean_df['anime_id'].isin(top_anime_ids)][['anime_id', 'Name', 'Genres', 'Score']].copy()
        est_dict = {pred.iid: pred.est for pred in top_preds}
        top_df['predicted_rating'] = top_df['anime_id'].map(est_dict)

        return top_df.sort_values(by='predicted_rating', ascending=False)
        
    # def initiate_collaborative_model_training(self) -> CollaborativeModellingArtifact:
    #     """
    #     Triggers local training via the trainer, then uploads artifacts to S3.
    #     Returns an artifact containing the final S3 paths.
    #     """
    #     try:
    #         logging.info("Initiating collaborative model training pipeline.")
            
    #         model_trainer_config = CollaborativeModellingConfig()
            
    #         trainer = CollaborativeModelingTrainer(
    #             config=model_trainer_config,
    #             data_ingestion_artifact=self.data_ingestion_artifact,
    #             data_cleaning_artifact=self.data_cleaning_artifact
    #         )

    #         local_artifact = trainer.initiate_model_training()

    #         timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    #         s3_model_dir = f"Artifacts/Model_Trainer/{timestamp}"

    #         s3_nn_model_path = f"{s3_model_dir}/{os.path.basename(local_artifact.neural_nets_model_name)}"
    #         s3_svd_model_path = f"{s3_model_dir}/{os.path.basename(local_artifact.svd_model_name)}"
    #         s3_nn_user_encoder_path = f"{s3_model_dir}/{os.path.basename(local_artifact.neural_nets_user_encoder)}"
    #         s3_nn_anime_encoder_path = f"{s3_model_dir}/{os.path.basename(local_artifact.neural_nets_anime_encoder)}"
            
    #         save_data_to_S3(local_artifact.neural_nets_model_name, s3_nn_model_path)
    #         save_data_to_S3(local_artifact.svd_model_name, s3_svd_model_path)
    #         save_data_to_S3(local_artifact.neural_nets_user_encoder, s3_nn_user_encoder_path)
    #         save_data_to_S3(local_artifact.neural_nets_anime_encoder, s3_nn_anime_encoder_path)
    #         logging.info("All collaborative model artifacts have been uploaded to S3.")

    #         s3_artifact = CollaborativeModellingArtifact(
    #             neural_nets_model_name=s3_nn_model_path,
    #             svd_model_name=s3_svd_model_path,
    #             neural_nets_user_encoder=s3_nn_user_encoder_path,
    #             neural_nets_anime_encoder=s3_nn_anime_encoder_path
    #         )
    #         logging.info("Collaborative model training pipeline completed.")
    #         return s3_artifact
            
    #     except Exception as e:
    #         logging.error(ExceptionHandler(e, sys))
    #         raise ExceptionHandler(e, sys)
    def initiate_collaborative_model_training(self) -> CollaborativeModellingArtifact:
        """
        Triggers the training pipeline which now handles S3 uploads internally.
        """
        try:
            logging.info("Initiating collaborative model training pipeline.")
            model_trainer_config = CollaborativeModellingConfig()
            
            trainer = CollaborativeModelingTrainer(
                config=model_trainer_config,
                data_ingestion_artifact=self.data_ingestion_artifact,
                data_cleaning_artifact=self.data_cleaning_artifact
            )
            # The trainer now returns an artifact with S3 paths.
            s3_artifact = trainer.initiate_model_training()
            return s3_artifact
            
        except Exception as e:
            logging.error(ExceptionHandler(e, sys))
            raise ExceptionHandler(e, sys)
        
if __name__ == "__main__":
    try:        
        logging.info("--- [TEST SETUP] Creating prerequisite artifacts ---")
        
        # This artifact should point to the location of your ratings CSV in S3.
        data_ingestion_artifact = DataIngestionArtifact(
            feature_store_anime_file_path="dummy/path_not_used_by_this_component.csv",
            feature_store_rating_file_path="Artifacts/Data_ingestion/06-14-2025_22-09-04/Anime_Ratings.csv"
        )
        
        # This artifact should point to the location of your cleaned anime metadata in S3.
        data_cleaning_artifact = DataCleaningArtifact(
            cleaned_anime_data="Artifacts/Data_Cleaning/06-16-2025_20-05-59/Cleaned_Anime_Description.csv"
        )
        
        
        logging.info("\n--- [TEST TRAINING] Initiating Collaborative Model Training ---")
        
        # Instantiate the component for training. We don't pass a model_trainer_artifact.
        training_instance = CollaborativeModelling(
            data_ingestion_artifact=data_ingestion_artifact,
            data_cleaning_artifact=data_cleaning_artifact
        )
        
        # This one line runs the entire training and S3 upload process.
        # It returns an artifact containing the S3 paths of the newly created models.
        s3_model_artifact = training_instance.initiate_collaborative_model_training()
        
        logging.info(f"Training complete. Models and encoders uploaded to S3.")
        logging.info(f"Generated S3 Artifact: {s3_model_artifact}")
        
        
        logging.info("\n--- [TEST INFERENCE] Getting Recommendations ---")
        
        # --- IMPORTANT ---
        # If you COMMENTED OUT the training section above, you MUST UNCOMMENT the
        # block below and provide the correct S3 paths to your pre-trained models.
        """
        logger.info("Using pre-existing S3 model paths for inference.")
        s3_model_artifact = CollaborativeModellingArtifact(
             neural_nets_model_name='Artifacts/Model_Trainer/YOUR_TIMESTAMP/nn_weights.weights.h5',
             svd_model_name='Artifacts/Model_Trainer/YOUR_TIMESTAMP/svd_model.pkl',
             neural_nets_user_encoder='Artifacts/Model_Trainer/YOUR_TIMESTAMP/nn_user_encoder.pkl',
             neural_nets_anime_encoder='Artifacts/Model_Trainer/YOUR_TIMESTAMP/nn_anime_encoder.pkl'
        )
        """

        # Instantiate the component for inference, passing the artifact with S3 model paths.
        inference_instance = CollaborativeModelling(
            data_ingestion_artifact=data_ingestion_artifact,
            data_cleaning_artifact=data_cleaning_artifact,
            model_trainer_artifact=s3_model_artifact  # Use the artifact from the training step
        )

        test_user_id = 1863  # A user from your notebook
        num_recommendations = 5
        
        logging.info(f"Getting Top-{num_recommendations} recommendations for User ID: {test_user_id}")

        # Get recommendations from the Neural Network model
        nn_recommendations = inference_instance.get_nn_recommendations(user_id=test_user_id, N=num_recommendations)
        print("\n--- NEURAL NETWORK RECOMMENDATIONS ---")
        if not nn_recommendations.empty:
            print(nn_recommendations)
        else:
            print(f"Could not generate NN recommendations for user {test_user_id}.")

        # Get recommendations from the SVD model
        svd_recommendations = inference_instance.get_svd_recommendations(user_id=test_user_id, N=num_recommendations)
        print("\n--- SVD RECOMMENDATIONS ---")
        if not svd_recommendations.empty:
            print(svd_recommendations)
        else:
            print(f"Could not generate SVD recommendations for user {test_user_id}.")
            
        logging.info("\n--- Collaborative Modelling Test Completed Successfully ---")

    except Exception as e:
        logging.error(ExceptionHandler(e, sys))
        raise ExceptionHandler(e, sys)