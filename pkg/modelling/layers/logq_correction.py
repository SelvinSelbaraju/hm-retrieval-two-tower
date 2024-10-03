from typing import Dict
import tensorflow as tf


class LogQCorrection(tf.keras.layers.Layer):
    """
    Apply the LogQ Correction to logits.
    Used for in-batch negative sampling.

    Parameters
    ----------
    candidate_prob_lookup: Dict[str, float]
        A dict mapping candidate ids to probs.
    """

    def __init__(
        self,
        candidate_prob_lookup: Dict[str, float],
    ):
        super().__init__()
        self._init_lookup(candidate_prob_lookup)

    def _init_lookup(self, candidate_prob_lookup: Dict[str, float]) -> None:
        """
        Create the lookup.

        Parameters
        ----------
        candidate_prob_lookup: Dict[str, float]
            A dict mapping candidate ids to probs.
        """
        self.lookup = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(candidate_prob_lookup.keys()),
                values=list(candidate_prob_lookup.values()),
                key_dtype=tf.string,
                value_dtype=tf.float32,
            ),
            # What to do if key not found
            # Ln(1) is 0, so subtract nothing
            default_value=1.0,
        )

    def __call__(
        self, logits: tf.Tensor, candidate_ids: tf.Tensor
    ) -> tf.Tensor:
        """
        Apply the logQ correction from the lookup.

        Parameters
        ----------
        logits: tf.Tensor
            Affinity scores for every query/candidate pair.
            Tensor has shape (B x B), B is batch_size.
        candidate_ids: tf.Tensor
            The candidate ids for each positive.
            Tensor has shape (B x 1).

        Returns
        -------
        results: tf.Tensor
            Tensor of logits with logQ subtracted.
        """
        # The candidates should be a 1 x B
        # Candidate are in the logits column axis
        candidate_ids = tf.transpose(candidate_ids)
        # Get the probs and apply the log
        corrections = tf.math.log(self.lookup.lookup(candidate_ids))
        # Corrections should be broadcast into B x B
        # The single row should be duplicated B times
        return logits - corrections
