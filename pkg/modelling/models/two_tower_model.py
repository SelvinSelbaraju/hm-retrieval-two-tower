import os
from typing import List, Optional, Dict
import logging
import tensorflow as tf
from pkg.modelling.models.tower import Tower
from pkg.schema.features import Feature
from pkg.schema.schema import Schema
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel
from pkg.modelling.layers.logq_correction import LogQCorrection


class TwoTowerModel(AbstractKerasModel):
    """
    Two Tower Model class.

    Parameters
    ----------
    query_features: List[str]
        Features for the query tower.
    candidate_features: List[str]
        Features for the candidate tower.
    candidate_id_col: str
        The column containing the candidate_id.
    joint_embedding_size: int
        Joint embedding size which gets dot product.
    query_tower_units: Optional[List[int]]
        Hidden units for the query tower.
    candidate_tower_units: Optional[List[int]]
        Hidden units for the candidate tower.
    candidate_prob_lookup: Optional[Dict[str, Float]]
        If provided, will perform logQ correction before passing to loss.
    """

    def __init__(
        self,
        query_features: List[Feature],
        candidate_features: List[Feature],
        candidate_id_col: str,
        joint_embedding_size: int,
        query_tower_units: Optional[List[int]] = None,
        candidate_tower_units: Optional[List[int]] = None,
        candidate_prob_lookup: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.query_features = query_features
        self.candidate_features = candidate_features
        if candidate_id_col not in [f.name for f in candidate_features]:
            raise ValueError(
                f"candidate_id_col {candidate_id_col} not a candidate feature"
            )
        self.candidate_id_col = candidate_id_col
        self.query_tower = Tower(
            query_features, joint_embedding_size, query_tower_units
        )
        self.candidate_tower = Tower(
            candidate_features, joint_embedding_size, candidate_tower_units
        )
        # Used to perform the logQ correction
        if candidate_prob_lookup:
            self.logq_correction = LogQCorrection(candidate_prob_lookup)
        else:
            self.logq_correction = None
        self.initialise_model()

    def call(
        self, x: Dict[str, tf.Tensor], training: bool = True
    ) -> tf.Tensor:
        """
        Pass inputs through the model.

        Parameters
        ----------
        x: Dict[str, tf.Tensor]
            Dict of features to tensors.
        training: bool
            Whether the model is in training model.
            Defaults to True.
        Returns
        -------
        outputs: tf.Tensor
            The score of the ith query with the jth candidate.
            (Q x C) tensor, Q is num_queries, C is num_candidates.
            In training, Q = C = B, B is the batch_size.
            This is because there is one query/candidate pair per row.
        """
        query_features = {f.name: x[f.name] for f in self.query_features}
        candidate_features = {
            f.name: x[f.name] for f in self.candidate_features
        }
        queries = self.query_tower(query_features)
        candidates = self.candidate_tower(candidate_features)
        return tf.linalg.matmul(queries, candidates, transpose_b=True)

    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, float]:
        """
        Custom train step to overwrite default behaviour.
        This allows to do in-batch negative sampling.

        Parameters
        ----------
        data: Dict[str, tf.Tensor]
            Data from a TF Dataset containing the features.
            Does not contain labels as each instance is a positive.

        Returns
        -------
        metrics: Dict[str, float]
            The metrics for all of the provided metrics in compile.
        """
        with tf.GradientTape() as tape:
            y_pred = self(data, training=True)
            # LogQ Correction
            if self.logq_correction:
                y_pred = self.logq_correction(
                    y_pred, data[self.candidate_id_col]
                )
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
    def create_from_schema(
        cls, schema: Schema, candidate_id_col: str
    ) -> "TwoTowerModel":
        """
        Class method to create an instance from a Schema obj.

        Parameters
        ----------
        schema: Schema
            Schema object with features and model config.
        candidate_id_col" str
            The column with the candidate id.
        Returns
        -------
        model: TwoTowerModel
            Instance of the TwoTowerModel class.
        """
        return TwoTowerModel(
            query_features=schema.query_features,
            candidate_features=schema.candidate_features,
            candidate_id_col=candidate_id_col,
            joint_embedding_size=schema.model_config.joint_embedding_size,
            query_tower_units=schema.model_config.query_tower_units,
            candidate_tower_units=schema.model_config.candidate_tower_units,
            candidate_prob_lookup=schema.training_config.candidate_prob_lookup,
        )

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        Given query, candidate features, return the input signature.

        Returns
        -------
        input_signature: Dict[str, tf.TensorSpec]
            A dict mapping features to TensorSpec objs.
        """
        input_signature = {}
        for f in self.candidate_features + self.query_features:
            input_signature[f.name] = tf.TensorSpec(
                shape=(None, 1), dtype=f.dtype, name=f.name
            )
        return input_signature

    def save(self, model_path: str) -> None:
        """
        Create a directory at the model_path.
        Save the two_tower model, and each of the towers separately.
        Saved in two_tower, query_tower, candidate_tower dirs
            respectively.

        Parameters
        ----------
        model_path: str
            The path to save the model at.
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
