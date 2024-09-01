from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """
    Hyperparameters for training

    Parameters
    ----------
    train_batch_size: int
        Number of rows in a single batch of data
    shuffle_size: Optional[int] = None
        Shuffle buffer size. If None, don't shuffle
    """
    train_batch_size: int
    shuffle_size: Optional[int] = None
