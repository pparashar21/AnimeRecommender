"""
File to store the data classes of the artifacts and their directory locations in S3 (saved as Key values only)
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact: 
    feature_store_anime_file_path:str
    feature_store_rating_file_path:str

@dataclass
class DataCleaningArtifact:
    # merged_data:str
    cleaned_anime_data:str
    
@dataclass
class DataTransformationArtifact:
    transformed_content_data:str

@dataclass
class ContentModellingArtifact:
    knn_model_name:str

@dataclass
class CollaborativeModellingArtifact:
    neural_nets_model_name:str
    svd_model_name:str
    neural_nets_user_encoder:str
    neural_nets_anime_encoder:str