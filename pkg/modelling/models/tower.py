from typing import Optional, List, Dict
import tensorflow as tf
from pkg.schema.features import Feature
from pkg.modelling.layers.input_layer import InputLayer
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel

class Tower(AbstractKerasModel):
    """
    Tower as a Simple feed forward network for a two tower model

    Parameters
    ----------
    features: List[Feature]
        List of feature objs for this tower
    joint_embedding_size: int
        Size used for taking the dot product
    hidden_units: Optional[List[int]]
        Optional hidden units in the tower
    """
    def __init__(
        self,
        features: List[Feature],
        joint_embedding_size: int,
        hidden_units: Optional[List[int]] = None
    ):
        super().__init__()
        self.features = features
        self.joint_embedding_size = joint_embedding_size
        self.hidden_units = hidden_units
        self._init_layers()
        self.initialise_model()
    

    def _init_layers(self) -> None:
        self.model_layers = [InputLayer(self.features)]
        if self.hidden_units:
            for units in self.hidden_units:
                self.model_layers.append(tf.keras.layers.Dense(units,activation="relu"))
        # Use simple linear layer before the dot product to prevent certain dims being 0
        self.model_layers.append(tf.keras.layers.Dense(self.joint_embedding_size,activation="linear"))
    
    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        output = x
        for layer in self.model_layers:
            output = layer(output)
        return output

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        input_signature = {}
        for f in self.features:
            input_signature[f.name] = tf.TensorSpec(shape=(None,1), dtype=f.dtype, name=f.name)
        return input_signature
