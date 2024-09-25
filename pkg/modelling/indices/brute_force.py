from typing import Tuple, Dict
import tensorflow as tf
from tensorflow.python.framework.tensor import TensorSpec
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel

class BruteForceIndex(AbstractKerasModel):
    """
    Store ids and embeddings for candidates
    At inference, return the top k ids for the query

    Parameters
    ----------
    candidate_model: Optional[tf.keras.Model]
        Optional model for embedding candidates
    k: int
        The number of results the index should return
    """
    def __init__(
        self,
        k: int,
        query_model: AbstractKerasModel,
        id_candidate_pairs: tf.data.Dataset
    ):
        super().__init__()
        self.k = k
        self.query_model = query_model
        self._index(id_candidate_pairs)
        self.initialise_model()
    
    def _index(self, id_candidate_pairs: tf.data.Dataset) -> None:
        """
        Creates a pseudo-model which returns the candidate candidates for a query
        This model is then used in the call method when passed candidates
        
        Parameters
        ----------
        id_candidate_pairs: tf.data.Dataset
            TF Dataset which yields tuples of (id,candidate_embedding)
        """
        identifiers, candidates = self.get_id_embeddings_from_dataset(id_candidate_pairs)
        # Since ids are strings, we can't use self.add_weight
        self._identifiers = identifiers
        self._candidates = self.add_weight(
            name="candidates",
            dtype=candidates.dtype,
            shape=candidates.shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )
        self._candidates.assign(candidates)
    
    def call(self, queries: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Return the top k candidates for a set of queries

        Parameters
        ----------
        queries: tf.Tensor
            Must be a dict of tensors for the query tower
        """
        query_embeddings = self.query_model(queries)
        scores = tf.linalg.matmul(query_embeddings, self._candidates, transpose_b=True)
        
        # The first output is the actual scores, we don't need those
        _, indices = tf.math.top_k(scores, k=self.k)
        # Looks up the indices in the tensor of identifiers
        return tf.gather(self._identifiers, indices)

    @staticmethod
    def get_id_embeddings_from_dataset(candidates: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        From a TF Dataset of (id,embedding), return them separately
        
        Parameters
        ----------
        candidates: tf.data.Dataset
            Returns tuples of (id,embedding) pairs
        """
        identifiers_and_candidates = list(candidates)
        candidates = tf.concat(
          [embeddings for _, embeddings in identifiers_and_candidates],
          axis=0
        )
        identifiers = tf.concat(
          [identifiers for identifiers, _ in identifiers_and_candidates],
          axis=0
        )
        return identifiers,candidates

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        return self.query_model.get_input_signature()

    
