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
    query_tower_units: List[int]
        Hidden units for the query tower
    candidate_tower_units: List[int]
        Hidden units for the candidate tower
    ks: List[int]
        The Recall@k metrics we want to evaluate
    """

    joint_embedding_size: int
    ks: List[int]
    query_tower_units: Optional[List[int]] = None
    candidate_tower_units: Optional[List[int]] = None
