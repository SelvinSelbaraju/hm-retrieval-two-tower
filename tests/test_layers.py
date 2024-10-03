from typing import Dict
import pytest
import tensorflow as tf
from pkg.modelling.layers.logq_correction import LogQCorrection


@pytest.fixture
def dummy_logits() -> tf.Tensor:
    return tf.constant(
        [[1.0, -1.5, 2.5], [-1.0, -2.5, 1.5], [2.5, -1.5, -1.0]],
        shape=(3, 3),
        dtype=tf.float32,
    )


@pytest.fixture
def candidate_ids() -> tf.Tensor:
    return tf.constant([["id1", "id2", "id3"]], shape=(3, 1), dtype=tf.string)


@pytest.fixture
def candidate_prob_lookup() -> Dict[str, float]:
    return {"id1": 0.3, "id2": 0.2, "id3": 0.5}


def test_logq_correction(dummy_logits, candidate_ids, candidate_prob_lookup):
    logq_layer = LogQCorrection(candidate_prob_lookup)
    expected = tf.constant(
        [
            [2.2039728043, 0.1094379124, 3.1931471806],
            [0.2039728043, -0.8905620876, 2.1931471806],
            [3.7039728043, 0.1094379124, -0.3068528194],
        ],
        shape=(3, 3),
        dtype=tf.float32,
    )
    result = logq_layer(dummy_logits, candidate_ids)
    # Round as float32 has limited precision
    tf.assert_equal(tf.round(result, 5), tf.round(expected, 5))
