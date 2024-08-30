from typing import Optional, List
import pandas as pd
import tensorflow as tf
# For now, Feature has a name, vocabulary, data type

class Feature:
    """
    Class which contains all of the information for an input feature

    Parameters
    ----------
    name: str
        name of the feature
    dtype: tf.dtypes.DType
        Tensorflow dtype, must be one of valid types
    vocab: Optional[List[str]]
        For categorical, values the feature can take
        Any new values seen will be set as unknown in the model
    max_vocab_size: Optional[int]
        Max size of vocab, useful for limiting the size
        Ignored if the vocab is provided up front
    """
    VALID_DTYPES = [tf.string, tf.float32]
    def __init__(
        self,
        name: str,
        dtype: tf.dtypes.DType,
        vocab: Optional[List[str]] = None,
        max_vocab_size: Optional[int] = None,
    ):
        self.name = name

        if dtype not in self.VALID_DTYPES:
            raise TypeError(f"dtype must be one of {self.VALID_DTYPES}, got {dtype}")
        else:
            self.dtype = dtype

        if self.vocab:
            self.vocab = set(vocab)
            self.is_built = True
        else:
            self.is_built = False
        
        if max_vocab_size:
            if not isinstance(max_vocab_size, int):
                raise TypeError(f"max_vocab_size must be an int, got {max_vocab_size}")
        self.max_vocab_size = max_vocab_size
        
    
    def set_vocab_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set the vocabulary from a Pandas Dataframe

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing data to build the vocab from
        """
        if self.name not in df.columns:
            raise ValueError(f"Feature name {self.name} not found in df cols {df.columns}")
        v_counts = df[self.name].value_counts()
        if self.max_vocab_size:
            self.vocab = list(v_counts.head(self.max_vocab_size).index)
        else:
            self.vocab = list(v_counts.index)
        self.is_built = True


