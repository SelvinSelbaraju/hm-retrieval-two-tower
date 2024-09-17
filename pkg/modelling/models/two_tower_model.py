from typing import List, Optional, Dict
import tensorflow as tf
from pkg.modelling.models.tower import Tower
from pkg.schema.features import Feature
from pkg.schema.schema import Schema

class TwoTowerModel(tf.keras.Model):
    """
    Two Tower Model class

    Parameters
    ----------
    user_features: List[str]
        Features for the user tower
    item_features: List[str]
        Features for the item tower
    joint_embedding_size: int
        Joint embedding size which gets dot product
    user_tower_units: Optional[List[int]]
        Hidden units for the user tower
    item_tower_units: Optional[List[int]]
        Hidden units for the item tower
    """
    def __init__(
        self,
        user_features: List[Feature],
        item_features: List[Feature],
        joint_embedding_size: int,
        user_tower_units: Optional[List[int]] = None,
        item_tower_units: Optional[List[int]] = None
    ):
        super().__init__()
        self.user_tower = Tower(user_features, joint_embedding_size, user_tower_units)
        self.item_tower = Tower(item_features, joint_embedding_size, item_tower_units)
    
    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        users = self.user_tower(x)
        items = self.item_tower(x)
        return tf.linalg.matmul(users, items, transpose_b=True)


    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, float]:
        """
        Custom train step to overwrite default behaviour
        This allows to do in-batch negative sampling

        Parameters
        ----------
        data: Dict[str, tf.Tensor]
            Data from a TF Dataset containing the features
            Does not contain labels as each instance is a positive

        Returns
        -------
        metrics: Dict[str, float]
            The metrics for all of the provided metrics in compile
        """
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            # B x B tensor
            # Diagonal contains scores for true positives
            labels = tf.eye(tf.shape(y_pred)[0],tf.shape(y_pred)[0], dtype=tf.float32)
            loss = self.compute_loss(y=labels, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def create_from_schema(cls, schema: Schema) -> "TwoTowerModel":
        return TwoTowerModel(
            user_features=[f for f in schema.features if f.name in schema.model_config.user_features],
            item_features=[f for f in schema.features if f.name in schema.model_config.item_features],
            joint_embedding_size=schema.model_config.joint_embedding_size,
            user_tower_units=schema.model_config.user_tower_units,
            item_tower_units=schema.model_config.item_tower_units
        )

    

