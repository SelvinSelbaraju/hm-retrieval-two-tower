from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """
    Config for a two-tower model

    Parameters
    ----------
    joint_embedding_size: int
        Joint embedding size which gets dot product
    user_tower_units: List[int]
        Hidden units for the user tower
    item_tower_units: List[int]
        Hidden units for the item tower
    ks: List[int]
        The Recall@k metrics we want to evaluate
    """
    joint_embedding_size: int
    ks: List[int]
    user_tower_units: Optional[List[int]] = None
    item_tower_units: Optional[List[int]] = None
