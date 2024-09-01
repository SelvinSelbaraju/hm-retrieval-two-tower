from typing import List
import logging
import os
import pickle
import pandas as pd
from pkg.schema.features import Feature
from pkg.schema.training_config import TrainingConfig

logger = logging.getLogger(__name__)

class Schema:
    """
    Schema which stores all config and stats for modelling
    
    Parameters
    ----------
    features: List[Feature]
        List of Feature objects for modelling
    training_config: TrainingConfig
        Training configuration
    """
    def __init__(self, features: List[Feature], training_config: TrainingConfig):
        self.features = features
        self.training_config = training_config
    
    def build_features_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Build all features from a Pandas Dataframe, eg. set the stats

        Parameters
        ----------
        df: pd.DataFrame
            Pandas Dataframe to set feature stats from
        """
        for feature in self.features:
            if not feature.is_built:
                feature.set_vocab_from_dataframe(df)
    
    def save(self, filepath: str) -> None:
        """
        Save the Schema as a pickle to be used later

        Parameters
        ----------
        filepath: str
            Full filepath to save the object at
        """
        logger.info(f"Saving Schema obj at filepath: {filepath}")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    

    @classmethod
    def load_from_filepath(cls, filepath: str) -> "Schema":
        logger.info(f"Loading Schema obj from {filepath}")
        with open(filepath, "rb") as f:
            schema = pickle.load(f)
        return schema
        

