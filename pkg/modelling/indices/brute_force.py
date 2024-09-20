from typing import Tuple, Optional, Union, Dict
import tensorflow as tf

class BruteForceIndex(tf.keras.Model):
    """
    Store ids and embeddings for items
    At inference, return the top k ids for the query

    Parameters
    ----------
    item_model: Optional[tf.keras.Model]
        Optional model for embedding items
    k: int
        The number of results the index should return
    """
    def __init__(
        self,
        k: int,
        user_model: tf.keras.Model
    ):
        super().__init__()
        self.k = k
        self.user_model = user_model
    
    def index(self, id_item_pairs: tf.data.Dataset) -> None:
        """
        Creates a pseudo-model which returns the candidate items for a query
        This model is then used in the call method when passed items
        
        Parameters
        ----------
        id_item_pairs: tf.data.Dataset
            TF Dataset which yields tuples of (id,item_embedding)
        """
        identifiers, candidates = self.get_id_embeddings_from_dataset(id_item_pairs)
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
    
    def call(self, users: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Return the top k candidates for a set of users

        Parameters
        ----------
        users: tf.Tensor
            Must be a dict of tensors for the user tower
        """
        if self.user_model:
            user_embeddings = self.user_model(users)
        scores = tf.linalg.matmul(user_embeddings, self._candidates, transpose_b=True)
        
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
    
