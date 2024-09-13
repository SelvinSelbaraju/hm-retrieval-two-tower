from typing import List, Dict
import tensorflow as tf
from pkg.schema.features import Feature

class InputLayer(tf.keras.layers.Layer):
    """
    Convert a dict of tensors, embed categorical and concat together

    Parameters
    ----------
    features: List[Feature]
        List of Feature objs containing tf dtypes
    """
    def __init__(
        self,
        features: List[Feature]
    ):
        super().__init__()
        self.numerical_features = [f for f in features if f.dtype != tf.string]
        self.categorical_features = [f for f in features if f.dtype == tf.string]
        self._init_embedding_layers()


    def _init_embedding_layers(self) -> None:
        self.embedding_layers = {}
        for f in self.categorical_features:
            self.embedding_layers[f.name] = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    num_oov_indices=1,
                    vocabulary=f.vocab,
                ),
                tf.keras.layers.Embedding(
                    len(f.vocab)+1, # add 1 for OOV
                    f.embedding_size,
                ),
                tf.keras.layers.Reshape((f.embedding_size,))
            ])
    
    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        inputs = []
        # Simply append numerical features
        for f in self.numerical_features:
            inputs.append(x[f.name])
        # Change strings to int and then lookup embeddings
        for f in self.categorical_features:
            inputs.append(self.embedding_layers[f.name](x[f.name]))
        inputs = tf.keras.layers.Concatenate()(inputs)
        return inputs
        
        

