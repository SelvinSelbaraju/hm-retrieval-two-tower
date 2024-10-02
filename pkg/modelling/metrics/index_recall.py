from typing import Dict, List, Union, Optional
import logging
import tensorflow as tf
from pkg.modelling.indices.brute_force import BruteForceIndex
from pkg.modelling.indices.static_index import StaticIndex

logger = logging.getLogger(__name__)


class IndexRecall:
    """
    Given an index, calculate a recall@K metric.

    Parameters
    ----------
    index: Union[BruteForceIndex, StaticIndex]
        The index to calculate metrics for.
    ks: List[int]
        The top Ks to calculate recall at.
    """

    def __init__(
        self,
        index: Union[BruteForceIndex, StaticIndex],
        ks: List[int],
    ):
        self.index = index
        self.ks = ks
        self.hits = {k: tf.constant(0, dtype=tf.int32) for k in ks}
        self.seen = tf.constant(0, dtype=tf.int32)
        self.metric = {k: tf.constant(0, dtype=tf.int32) for k in ks}

    def __call__(
        self, queries: Dict[str, tf.Tensor], true_candidate_ids: tf.Tensor
    ) -> Dict[int, float]:
        """
        Given queries and the true candidate ids, return the recall

        Parameters
        ----------
        queries: Dict[str, tf.Tensor]
            Dict of query features to their values.
        true_candidate_ids: tf.Tensor
            Tensor containing the true candidate for each query.
            Shape should be (num_queries,).

        Returns
        -------
        metric: Dict[int, float]
            A dict with K as the key, and recall@K.
        """
        candidates = self.index(queries)
        self.seen += true_candidate_ids.shape[0]
        for k in self.ks:
            recall = tf.math.equal(true_candidate_ids, candidates[:, :k])
            recall = tf.reduce_sum(tf.cast(recall, tf.int32))
            self.hits[k] += recall
            self.metric[k] = self.hits[k] / self.seen
        return self.metric

    def log_metric(
        self, epoch: Optional[int] = None, to_tensorboard: bool = True
    ) -> None:
        """
        Log all of the recall@K values to Tensorboard.

        Parameters
        ----------
        epoch: Optional[int]
            Epoch this metric is for. Used for logging.
        to_tensorboard: bool
            Whether to log the metric to Tensorboard.
            Defaults to True.
        """
        for k in self.ks:
            logger.info(
                f"Start of epoch {epoch} recall@{k}: {self.metric[k].numpy()}"
            )
            if to_tensorboard:
                tf.summary.scalar(
                    f"Epoch Start Recall@{k}",
                    data=self.metric[k].numpy(),
                    step=epoch,
                )
