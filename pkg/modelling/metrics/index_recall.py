from typing import Dict, List
import logging
import tensorflow as tf
from pkg.modelling.indices.brute_force import BruteForceIndex

logger = logging.getLogger(__name__)

class IndexRecall:
    """
    Given an index, calculate a recall metric

    Parameters
    ----------
    index: BruteForceIndex
        The index to calculate metrics for
    ks: List[int]
        The top ks to calculate recall at
    """
    def __init__(
        self,
        index: BruteForceIndex,
        ks: List[int],
    ):
        self.index = index
        self.ks = ks
        if max(ks) > index.k:
            logger.info(f"Overwriting index k to {max(ks)}")
            index.k = max(ks)
        self.max_k = max(ks)
        self.hits = {k: tf.constant(0, dtype=tf.int32) for k in ks}
        self.seen =  tf.constant(0, dtype=tf.int32)
        self.metric = {k: tf.constant(0, dtype=tf.int32) for k in ks}
    

    def __call__(self, queries: Dict[str,tf.Tensor], true_candidate_ids: tf.Tensor) -> float:
        """
        Given queries and the true candidate ids, return the recall

        Parameters
        ----------
        queries: Dict[str,tf.Tensor]
            Dict of query features to their values
        true_candidate_ids: tf.Tensor
            Tensor containing the true candidate for each query
            Shape should be [num_queries,1]
        """
        candidates = self.index(queries)
        self.seen += true_candidate_ids.shape[0] 
        for k in self.ks:
            recall = tf.math.equal(true_candidate_ids, candidates[:,:k])
            recall = tf.reduce_sum(tf.cast(recall, tf.int32))
            self.hits[k] += recall
            self.metric[k] = self.hits[k] / self.seen
        return self.metric


    def log_to_tensorboard(self, epoch: int) -> None:
        """
        Log all of the k values to Tensorboard
        """
        for k in self.ks:
            logger.info(f"Start of epoch {epoch} recall@{k}: {self.metric[k].numpy()}")
            tf.summary.scalar(f"Epoch Start Recall@{k}", data=self.metric[k].numpy(), step=epoch)

    