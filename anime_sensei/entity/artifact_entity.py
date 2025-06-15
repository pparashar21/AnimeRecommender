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
class DataTransformationArtifact:
    merged_data:str