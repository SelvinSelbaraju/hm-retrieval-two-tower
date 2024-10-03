import pytest
from typing import Dict, Tuple
import tensorflow as tf
from pkg.modelling.models.abstract_keras_model import AbstractKerasModel
from pkg.modelling.indices.brute_force import BruteForceIndex


class MockEmbeddingModel(AbstractKerasModel):
    """
    Mock model that embeds strings.
    Inputs to model are dict with the key id.
    Value is correpsonding tf.string tensor.

    Parameters
    ----------
    ids: tf.Tensor
        A tf.string tensor of ids.
        Has shape (K,), K is the num of keys.
    embeddings: tf.Tensor
        For each key, the corresponding embedding.
        Has shape (K, E), where E is the embed dim.
    """

    def __init__(self, ids: tf.Tensor, embeddings: tf.Tensor):
        super().__init__()
        self.string_lookup = tf.keras.layers.StringLookup(vocabulary=ids)
        self.embeddings = embeddings

    def call(self, x: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Given a dict of key id to a tensor of strings,
        return the embeddings for those strings.

        Parameters
        ----------
        x: Dict[str, tf.Tensor]
            Dict of {id: tensor} to embed.
            Tensor has shape (B x 1), B is batch_size.

        Returns
        -------
        results: tf.Tensor
            The embedded strings.
            Has shape (B x E).
        """
        indices = self.string_lookup(x["id"])
        return tf.gather_nd(self.embeddings, indices)

    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        Create the dummy input signature.

        Returns
        -------
        input_signature: Dict[str, tf.TensorSpec]
            A dict mapping features to TensorSpec objs.
        """
        return {
            "id": tf.TensorSpec(name="id", dtype=tf.string, shape=(None, 1))
        }


@pytest.fixture
def dummy_embeddings() -> Tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.constant(
            [
                "query_1",
                "query_2",
                "query_3",
            ],
            shape=(3,),
            dtype=tf.string,
        ),
        tf.constant(
            [[1.0, 1.0], [0.5, -1.0], [1.0, -0.5], [-1.0, -0.5]],
            shape=(4, 2),
            dtype=tf.float32,
        ),
    )


@pytest.fixture
def id_candidate_pairs() -> tf.data.Dataset:
    pairs = (
        [
            "candidate_1",
            "candidate_2",
            "candidate_3",
            "candidate_4",
            "candidate_5",
        ],
        [
            tf.constant([2.0, -1.5], dtype=tf.float32, shape=(2,)),
            tf.constant([-1.5, 3.0], dtype=tf.float32, shape=(2,)),
            tf.constant([-0.5, -1.0], dtype=tf.float32, shape=(2,)),
            tf.constant([1.0, -1.5], dtype=tf.float32, shape=(2,)),
            tf.constant([-2.0, -1.5], dtype=tf.float32, shape=(2,)),
        ],
    )
    ds = tf.data.Dataset.from_tensor_slices(pairs)
    return ds.batch(1)


def test_brute_force_index(dummy_embeddings, id_candidate_pairs):
    dummy_query_model = MockEmbeddingModel(
        dummy_embeddings[0], dummy_embeddings[1]
    )
    inputs = {
        "id": tf.constant(
            ["query_1", "query_2", "query_3", "query_4", "query_1"],
            shape=(5, 1),
            dtype=tf.string,
        )
    }
    # This was calculated by hand
    # Done by dot product each query/candidate pair
    # And seeing highest scores
    expected = tf.constant(
        [
            ["candidate_1", "candidate_4"],
            ["candidate_1", "candidate_4"],
            ["candidate_5", "candidate_3"],
            ["candidate_2", "candidate_1"],
            ["candidate_1", "candidate_4"],
        ],
        shape=(5, 2),
        dtype=tf.string,
    )
    index = BruteForceIndex(2, dummy_query_model, id_candidate_pairs)
    result = index(inputs)
    tf.assert_equal(expected, result)
