from typing import List, Dict
import pandas as pd
import tensorflow as tf
from pkg.schema.schema import Schema
from pkg.schema.features import Feature
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel


class StaticIndex(AbstractKerasModel):
    """
    Return a fixed set of candidates in order.
    Useful for building rules-based baselines.

    Parameters
    ----------
    k: int
        The number of candidates to return.
    input_features: List[Feature]
        The input features to this model.
        They are not used to generate candidates.
        But are used to follow the inputs of other indices.
        The batch size is used to duplicate predictions for each row.
    candidates: tf.Tensor
        Ordered candidate ids (tf.string) to return.
        Expected shape is (1,num_candidates).
    """

    def __init__(
        self, k: int, input_features: List[Feature], candidates: tf.Tensor
    ):
        super().__init__()
        self.k = k
        self.input_features = input_features
        self.candidates = candidates
        self.initialise_model()

    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Return the top K candidates,
        duplicated with batch size.

        Parameters
        ----------
        x: Dict[str, tf.Tensor]
            Dict of input name as key, tf.Tensor as values.
        """
        num_results = tf.shape(x[self.input_features[0].name])[0]
        return tf.tile(self.candidates[:, : self.k], (num_results, 1))

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        Fetch the input signature for the index.
        Used for saving in a servable format.
        """
        return {
            f.name: tf.TensorSpec(shape=(None, 1), dtype=f.dtype, name=f.name)
            for f in self.input_features
        }

    @classmethod
    def build_popularity_index_from_series_schema(
        cls, schema: Schema, s: pd.Series
    ) -> "StaticIndex":
        """
        Given the candidate id col from a df of transactions,
        build the index based on candidate popularity.

        Parameters
        ----------
        schema: Schema
            Schema for modelling.
        s: pd.Series
            The column of the transactions df to build the index.
        """
        ids = s.value_counts().index
        ids_tensor = tf.constant(
            [str(id) for id in ids], dtype=tf.string, shape=(1, len(ids))
        )
        return cls(
            k=max(schema.model_config.ks),
            input_features=schema.query_features,
            candidates=ids_tensor,
        )
