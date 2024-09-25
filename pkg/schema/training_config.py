from dataclasses import dataclass
from typing import Optional, Dict
import tensorflow as tf


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
    optimizer: tf.keras.optimizers.Optimizer
        Optimizer to use
    candidate_batch_size: int
        Batch size for indexing candidates
    shuffle_size: Optional[int] = None
        Shuffle buffer size. If None, don't shuffle
    epochs: int = 1
        Number of training rounds
    candidate_prob_lookup: Optional[Dict[str, Float]]
        Optional lookup for logQ Correction
    """

    train_batch_size: int
    test_batch_size: int
    optimizer: tf.keras.optimizers.Optimizer
    candidate_batch_size: int = 10000
    shuffle_size: Optional[int] = None
    epochs: int = 1
    candidate_prob_lookup: Optional[Dict[str, float]] = None
