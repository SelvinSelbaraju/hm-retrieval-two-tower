from typing import Optional, List
from enum import Enum
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class FeatureFamily(Enum):
    """
    The families which features can be part of.
    Features can either be query features or candidate features.
    """

    QUERY = "query"
    CANDIDATE = "candidate"


class Feature:
    """
    Class which contains all of the information for an input feature.

    Parameters
    ----------
    name: str
        Name of the feature.
    dtype: tf.dtypes.DType
        Tensorflow dtype, must be one of valid types.
    feature_family: FeatureFamily
        Must be one of the families specified in the FeatureFamily enum.
    embedding_size: Optional[int]
        For categorical, embedding dimension.
    vocab: Optional[List[str]]
        For categorical, values the feature can take.
        Any new values seen will be set as unknown in the model.
    max_vocab_size: Optional[int]
        Max size of vocab, useful for limiting the size.
        Ignored if the vocab is provided up front.
    """

    VALID_DTYPES = [tf.string, tf.float32]

    def __init__(
        self,
        name: str,
        dtype: tf.dtypes.DType,
        feature_family: FeatureFamily,
        embedding_size: Optional[int] = None,
        vocab: Optional[List[str]] = None,
        max_vocab_size: Optional[int] = None,
    ):
        self.name = name
        if dtype not in self.VALID_DTYPES:
            raise TypeError(
                f"dtype must be one of {self.VALID_DTYPES}, got {dtype}"
            )
        self.dtype = dtype

        if feature_family not in FeatureFamily:
            raise ValueError(
                f"feature_family {feature_family} not valid. "
                f"Must be one of {FeatureFamily._member_names_}"
            )
        self.feature_family = feature_family

        if embedding_size:
            if dtype != tf.string:
                raise TypeError(
                    f"Got embedding size, dtype must be tf.string got {dtype}"
                )
        self.embedding_size = embedding_size
        self._init_vocab(vocab)

        if max_vocab_size:
            if not isinstance(max_vocab_size, int):
                raise TypeError(
                    f"max_vocab_size must be an int, got {max_vocab_size}"
                )
        self.max_vocab_size = max_vocab_size

    def _init_vocab(self, vocab: Optional[List[str]] = None) -> None:
        """
        Check the feature's dtype and init the vocab.

        Parameters
        ----------
        vocab: Optional[List[str]] = None
            The vocab passed to the feature's init method.
        """
        if self.dtype != tf.string:
            logger.info(
                f"Ignoring vocab passed for non-string feature {self.name}"
            )
            self.vocab = None
            self.is_built = True
        else:
            if vocab:
                self.vocab = set(vocab)
                self.is_built = True
            else:
                self.vocab = None
                self.is_built = False

    def set_vocab_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the vocabulary from a Pandas Dataframe.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing data to build the vocab from.
        """
        if self.name not in df.columns:
            raise ValueError(
                f"Feature name {self.name} not found in df cols {df.columns}"
            )
        v_counts = df[self.name].value_counts()
        if self.max_vocab_size:
            self.vocab = list(v_counts.head(self.max_vocab_size).index)
        else:
            self.vocab = list(v_counts.index)
        # The vocab must all be strings
        # Numpy array makes StringLookup layer MUCH faster vs list
        self.vocab = np.array([str(x) for x in self.vocab])
        self.is_built = True
