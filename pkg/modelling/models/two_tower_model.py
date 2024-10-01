import os
from typing import List, Optional, Dict
import logging
import tensorflow as tf
from pkg.modelling.models.tower import Tower
from pkg.schema.features import Feature
from pkg.schema.schema import Schema
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel


class TwoTowerModel(AbstractKerasModel):
    """
    Two Tower Model class

    Parameters
    ----------
    query_features: List[str]
        Features for the query tower
    candidate_features: List[str]
        Features for the candidate tower
    joint_embedding_size: int
        Joint embedding size which gets dot product
    query_tower_units: Optional[List[int]]
        Hidden units for the query tower
    candidate_tower_units: Optional[List[int]]
        Hidden units for the candidate tower
    candidate_prob_lookup: Optional[Dict[str, Float]]
        If provided, will perform logQ correction before passing to loss
    """

    def __init__(
        self,
        query_features: List[Feature],
        candidate_features: List[Feature],
        joint_embedding_size: int,
        query_tower_units: Optional[List[int]] = None,
        candidate_tower_units: Optional[List[int]] = None,
        candidate_prob_lookup: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.query_features = query_features
        self.candidate_features = candidate_features
        self.query_tower = Tower(
            query_features, joint_embedding_size, query_tower_units
        )
        self.candidate_tower = Tower(
            candidate_features, joint_embedding_size, candidate_tower_units
        )
        # Used to perform the logQ correction
        if candidate_prob_lookup:
            self.candidate_prob_lookup = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=list(candidate_prob_lookup.keys()),
                    values=list(candidate_prob_lookup.values()),
                    key_dtype=tf.string,
                    value_dtype=tf.float32,
                ),
                default_value=0.0,
                name="candidate_sampling_probs",
            )
        else:
            self.candidate_prob_lookup = None
        self.initialise_model()

    def call(
        self, x: Dict[str, tf.Tensor], training: bool = True
    ) -> tf.Tensor:
        query_features = {f.name: x[f.name] for f in self.query_features}
        candidate_features = {
            f.name: x[f.name] for f in self.candidate_features
        }
        queries = self.query_tower(query_features)
        candidates = self.candidate_tower(candidate_features)
        return tf.linalg.matmul(queries, candidates, transpose_b=True)

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
                corrections = self.candidate_prob_lookup.lookup(
                    data["article_id"]
                )
                # Match the shape of the B x B scores tensor
                corrections = tf.transpose(corrections)
                y_pred -= tf.math.log(corrections)
            # B x B tensor
            # Diagonal contains scores for true positives
            labels = tf.eye(
                tf.shape(y_pred)[0], tf.shape(y_pred)[0], dtype=tf.float32
            )
            loss = self.compute_loss(y=labels, y_pred=y_pred)

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(labels, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @classmethod
    def create_from_schema(cls, schema: Schema) -> "TwoTowerModel":
        return TwoTowerModel(
            query_features=schema.query_features,
            candidate_features=schema.candidate_features,
            joint_embedding_size=schema.model_config.joint_embedding_size,
            query_tower_units=schema.model_config.query_tower_units,
            candidate_tower_units=schema.model_config.candidate_tower_units,
            candidate_prob_lookup=schema.training_config.candidate_prob_lookup,
        )

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        input_signature = {}
        for f in self.candidate_features + self.query_features:
            input_signature[f.name] = tf.TensorSpec(
                shape=(None, 1), dtype=f.dtype, name=f.name
            )
        return input_signature

    def save(self, model_path: str) -> None:
        """
        Create a directory at the model_path
        Save the two_tower model, and each of the towers separately

        Parameters
        ----------
        model_path: str
            The path to save the model at
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        two_tower_model_path = os.path.join(
            os.path.dirname(model_path), "two_tower"
        )
        query_tower_model_path = os.path.join(
            os.path.dirname(model_path), "query_tower"
        )
        candidate_tower_model_path = os.path.join(
            os.path.dirname(model_path), "candidate_tower"
        )
        logging.info(f"Saving two tower model at path: {two_tower_model_path}")
        tf.saved_model.save(self, two_tower_model_path)
        logging.info(f"Saving query tower at path: {query_tower_model_path}")
        tf.saved_model.save(self.query_tower, query_tower_model_path)
        logging.info(
            f"Saving candidate tower at path: {candidate_tower_model_path}"
        )
        tf.saved_model.save(self.candidate_tower, candidate_tower_model_path)
