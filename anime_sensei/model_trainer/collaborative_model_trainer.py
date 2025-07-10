import os
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Surprise library for SVD
from surprise import Dataset, Reader, SVD

# Keras imports for Neural Network
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten, Add, Dense, Dropout #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore

# Custom project imports
from anime_sensei.utils.utility import save_object, read_file_from_S3, save_object_to_s3
from anime_sensei.entity.config_entity import CollaborativeModellingConfig
# CHANGED: Import the correct source artifacts
from anime_sensei.entity.artifact_entity import DataIngestionArtifact, DataCleaningArtifact, CollaborativeModellingArtifact
from anime_sensei.loggers.logging import logging

# The RecommenderNet function definition remains the same
def RecommenderNet(num_users, num_animes, embedding_size=64, dropout_rate=0.3):
    user_input = Input(shape=(1,), name='user_ratings_input')
    anime_input = Input(shape=(1,), name='anime_input')
    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_ratings_embedding', embeddings_regularizer=l2(1e-6))(user_input)
    anime_embedding = Embedding(input_dim=num_animes, output_dim=embedding_size, name='anime_embedding', embeddings_regularizer=l2(1e-6))(anime_input)
    user_bias = Embedding(input_dim=num_users, output_dim=1, name='user_ratings_bias')(user_input)
    anime_bias = Embedding(input_dim=num_animes, output_dim=1, name='anime_bias')(anime_input)
    dot_product = Dot(axes=-1)([user_embedding, anime_embedding])
    dot_product = Flatten()(dot_product)
    user_bias_flat = Flatten()(user_bias)
    anime_bias_flat = Flatten()(anime_bias)
    interaction = Add()([dot_product, user_bias_flat, anime_bias_flat])
    dense = Dense(128, activation='relu')(interaction)
    dense = Dropout(dropout_rate)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(dropout_rate)(dense)
    output = Dense(1, activation='linear')(dense)
    model = Model(inputs=[user_input, anime_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae', 'mse'])
    return model

class CollaborativeModelingTrainer:
    """
    Handles the training process and saves models locally.
    """
    def __init__(self, config: CollaborativeModellingConfig, 
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_cleaning_artifact: DataCleaningArtifact):
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_cleaning_artifact = data_cleaning_artifact
        
    def _get_full_anime_data(self) -> pd.DataFrame:
        logging.info("Loading and merging datasets for training.")
        clean_df_path = self.data_cleaning_artifact.cleaned_anime_data
        ratings_df_path = self.data_ingestion_artifact.feature_store_rating_file_path
        
        clean_df = read_file_from_S3(clean_df_path)
        ratings_df = read_file_from_S3(ratings_df_path)
        
        anime_full = pd.merge(clean_df, ratings_df, on='anime_id', suffixes=("_anime", "_user"), how='inner')
        if 'Anime Title' in anime_full.columns:
            anime_full.drop(['Anime Title'], axis=1, inplace=True)
            
        return anime_full

    def _get_lr_callback(self, *args, **kwargs):
        def lrfn(epoch):
            start_lr=1e-5; max_lr=5e-5; min_lr=1e-5; rampup_epochs=5; sustain_epochs=0; exp_decay=0.8
            if epoch < rampup_epochs: return (max_lr - start_lr) / rampup_epochs * epoch + start_lr
            elif epoch < rampup_epochs + sustain_epochs: return max_lr
            else: return (max_lr - min_lr) * exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr
        return LearningRateScheduler(lrfn, verbose=1)

    def train_nn_model(self):
        """Trains and saves the Neural Network model and its encoders locally."""
        logging.info("Starting Neural Network model training.")
        
        anime_full = self._get_full_anime_data()
        anime_full = shuffle(anime_full, random_state=42)

        user_encoder = LabelEncoder()
        anime_full["user_encoded"] = user_encoder.fit_transform(anime_full["user_id"])
        num_users = len(user_encoder.classes_)
        
        anime_encoder = LabelEncoder()
        anime_full["anime_encoded"] = anime_encoder.fit_transform(anime_full["anime_id"])
        num_animes = len(anime_encoder.classes_)
        
        save_object_to_s3(self.config.neural_nets_user_encoding_path, user_encoder)
        save_object_to_s3(self.config.neural_nets_anime_encoding_path, anime_encoder)
        logging.info("User and Anime encoders for NN saved directly to S3.")

        X = anime_full[['user_encoded', 'anime_encoded']].values
        y = anime_full["rating"].values
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_array = [X_train[:, 0], X_train[:, 1]]

        model = RecommenderNet(num_users, num_animes)
        
        os.makedirs(os.path.dirname(self.config.neural_nets_model_path), exist_ok=True)
        checkpoint = ModelCheckpoint(filepath=self.config.neural_nets_model_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)
        lr_scheduler = self._get_lr_callback()
        
        model.fit(X_train_array, y_train, batch_size=10000, epochs=2, validation_split=0.1, callbacks=[checkpoint, lr_scheduler, early_stopping], verbose=1)
        
        save_object_to_s3(self.config.neural_nets_model_path, self.config.neural_nets_model_path)
        logging.info("Neural Network model weights uploaded to S3.")

    def train_svd_model(self):
        """Trains and saves the SVD model locally."""
        logging.info("Starting SVD model training.")
        
        anime_full = self._get_full_anime_data()
        df = anime_full[['user_id', 'anime_id', 'rating']].copy()
        
        reader = Reader(rating_scale=(df.rating.min(), df.rating.max()))
        data = Dataset.load_from_df(df, reader)
        trainset = data.build_full_trainset()
        
        svd_model = SVD(n_factors=100, biased=True, verbose=True)
        svd_model.fit(trainset)
        
        # os.makedirs(os.path.dirname(self.config.svd_model_path), exist_ok=True)
        # save_object(self.config.svd_model_path, svd_model)
        # logging.info(f"SVD model trained and saved locally to {self.config.svd_model_path}.")
        save_object_to_s3(self.config.svd_model_path, svd_model)
        logging.info(f"SVD model trained and saved directly to S3.")

    def initiate_model_training(self) -> CollaborativeModellingArtifact:
        """Orchestrates the local training of both collaborative models."""
        self.train_nn_model()
        self.train_svd_model()
        
        model_trainer_artifact = CollaborativeModellingArtifact(
            neural_nets_model_name=self.config.neural_nets_model_path,
            svd_model_name=self.config.svd_model_path,
            neural_nets_user_encoder=self.config.neural_nets_user_encoding_path,
            neural_nets_anime_encoder=self.config.neural_nets_anime_encoding_path
        )
        logging.info("Local collaborative model training completed.")
        return model_trainer_artifact