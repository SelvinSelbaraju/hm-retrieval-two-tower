from typing import Dict
import tensorflow as tf
from pkg.modelling.indices.brute_force import BruteForceIndex

class IndexRecall:
    """
    Given an index, calculate a recall metric

    Parameters
    ----------
    index: BruteForceIndex
        The index to calculate metrics for
    """
    def __init__(
        self,
        index: BruteForceIndex,
    ):
        self.index = index
        self.hits = tf.constant(0, dtype=tf.int32)
        self.seen =  tf.constant(0, dtype=tf.int32)
        self.metric = None
    

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
        recall = tf.math.equal(true_candidate_ids, candidates)
        recall = tf.reduce_sum(tf.cast(recall, tf.int32))
        self.hits += recall
        self.seen += true_candidate_ids.shape[0]
        self.metric = self.hits / self.seen
        return self.metric

    