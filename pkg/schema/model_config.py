from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """
    Config for a two-tower model

    Parameters
    ----------
    user_features: List[str]
        Features for the user tower
    item_features: List[str]
        Features for the item tower
    joint_embedding_size: int
        Joint embedding size which gets dot product
    user_tower_units: List[int]
        Hidden units for the user tower
    item_tower_units: List[int]
        Hidden units for the item tower
    """
    user_features: List[str]
    item_features: List[str]
    joint_embedding_size: int
    user_tower_units: Optional[List[int]] = None
    item_tower_units: Optional[List[int]] = None
