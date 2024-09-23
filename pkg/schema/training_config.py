from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class TrainingConfig:
    """
    Hyperparameters for training

    Parameters
    ----------
    train_batch_size: int
        Number of rows in a single batch of train data
    test_batch_size: int
        Number of rows in a single batch of test data
    shuffle_size: Optional[int] = None
        Shuffle buffer size. If None, don't shuffle
    epochs: int = 1
        Number of training rounds
    candidate_prob_lookup: Optional[Dict[str, Float]]
        Optional lookup for logQ Correction
    """
    train_batch_size: int
    test_batch_size: int
    shuffle_size: Optional[int] = None
    epochs: int = 1
    candidate_prob_lookup: Optional[Dict[str, float]] = None
