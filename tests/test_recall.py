import pytest
import tensorflow as tf
from pkg.schema.features import Feature, FeatureFamily
from pkg.modelling.indices.static_index import StaticIndex
from pkg.modelling.metrics.index_recall import IndexRecall


@pytest.fixture
def dummy_test_ds():
    """
    Dummy tf dataset to mock test data
    StaticIndex returns id1 to id10 in order
    Dataset set so hits@1 = 1, hits@2 = 3
    """
    data_dict = {
        "query_id": tf.constant(
            [
                b"query1",
                b"query2",
                b"query3",
                b"query4",
                b"query5",
            ],
            shape=(5, 1),
            dtype=tf.string,
        ),
        "candidate_id": tf.constant(
            [
                b"id1",
                b"id7",
                b"id2",
                b"id2",
                b"id10",
            ],
            shape=(5, 1),
            dtype=tf.string,
        ),
    }
    ds = tf.data.Dataset.from_tensor_slices(data_dict)
    return ds.batch(2)


@pytest.fixture
def simple_static_index():
    """
    Return id1 to id10 in order
    """
    static_candidates = tf.constant(
        [
            b"id1",
            b"id2",
            b"id3",
            b"id4",
            b"id5",
            b"id6",
            b"id7",
            b"id8",
            b"id9",
            b"id10",
        ],
        shape=(1, 10),
        dtype=tf.string,
    )
    k = 5
    input_features = [
        Feature(
            name="query_id",
            dtype=tf.string,
            feature_family=FeatureFamily.QUERY,
            embedding_size=2,
        )
    ]
    index = StaticIndex(
        k=k, input_features=input_features, candidates=static_candidates
    )
    return index


def test_recall_calculation(simple_static_index, dummy_test_ds):
    """
    Given a static index which returns candidates, and true data,
    Assert the recall metric is calculated correctly
    """
    metric = IndexRecall(simple_static_index, ks=[1, 2, 5])
    for batch in dummy_test_ds:
        metric({"query_id": batch["query_id"]}, batch["candidate_id"])
    expected = {
        1: tf.constant(0.2, dtype=tf.float64),
        2: tf.constant(0.6, dtype=tf.float64),
    }
    for k in expected:
        assert (
            metric.metric[k] == expected[k]
        ), f"Expected {expected[k]}, got {metric.metric[k]}"
