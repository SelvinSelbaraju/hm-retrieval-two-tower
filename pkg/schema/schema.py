from typing import List
import pickle
import pandas as pd
from pkg.schema.features import Feature
from pkg.utils.logger import logger

class Schema:
    """
    Schema which stores all config and stats for modelling
    
    Parameters
    ----------
    features: List[Feature]
        List of Feature objects for modelling
    """
    def __init__(self, features: List[Feature]):
        self.features = features
    
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
    
    def save(self, filepath: str):
        """
        Save the Schema as a pickle to be used later

        Parameters
        ----------
        filepath: str
            Full filepath to save the object at
        """
        logger.info(f"Saving Schema obj at filepath: {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        

