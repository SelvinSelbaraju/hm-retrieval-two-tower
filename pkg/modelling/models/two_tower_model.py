import os
from typing import List, Optional, Dict
import logging
import tensorflow as tf
from tensorflow.python.framework.tensor import TensorSpec
from pkg.modelling.models.tower import Tower
from pkg.schema.features import Feature
from pkg.schema.schema import Schema
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel

class TwoTowerModel(AbstractKerasModel):
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
    candidate_prob_lookup: Optional[Dict[str, Float]]
        If provided, will perform logQ correction before passing to loss
    """
    def __init__(
        self,
        user_features: List[Feature],
        item_features: List[Feature],
        joint_embedding_size: int,
        user_tower_units: Optional[List[int]] = None,
        item_tower_units: Optional[List[int]] = None,
        candidate_prob_lookup: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.user_tower = Tower(user_features, joint_embedding_size, user_tower_units)
        self.item_tower = Tower(item_features, joint_embedding_size, item_tower_units)
        # Used to perform the logQ correction
        if candidate_prob_lookup:
            self.candidate_prob_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=list(candidate_prob_lookup.keys()),
                    values=list(candidate_prob_lookup.values()),
                    key_dtype=tf.string,
                    value_dtype=tf.float32
                ),
                default_value=0.0,
                name="candidate_sampling_probs"
            )
        else:
            self.candidate_prob_lookup = None
        self.initialise_model()
    
    def call(self, x: Dict[str, tf.Tensor], training: bool = True) -> tf.Tensor:
        user_features = {f.name: x[f.name] for f in self.user_features}
        item_features = {f.name: x[f.name] for f in self.item_features}
        users = self.user_tower(user_features)
        items = self.item_tower(item_features)
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
            # LogQ Correction
            if self.candidate_prob_lookup:
                corrections = self.candidate_prob_lookup.lookup(data["article_id"])
                # Match the shape of the B x B scores tensor
                corrections = tf.transpose(corrections)
                y_pred -= tf.math.log(corrections)
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
            user_features=schema.user_features,
            item_features=schema.item_features,
            joint_embedding_size=schema.model_config.joint_embedding_size,
            user_tower_units=schema.model_config.user_tower_units,
            item_tower_units=schema.model_config.item_tower_units,
            candidate_prob_lookup=schema.training_config.candidate_prob_lookup
        )
    
    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        input_signature = {}
        for f in (self.item_features + self.user_features):
            input_signature[f.name] = tf.TensorSpec(shape=(None,1), dtype=f.dtype, name=f.name)
        return input_signature

    

