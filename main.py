import tensorflow as tf
from pkg.schema.model_config import ModelConfig
from pkg.schema.schema import Schema
from pkg.schema.training_config import TrainingConfig
from pkg.schema.features import Feature, FeatureFamily
from pkg.utils.settings import Settings
from pkg.etl.runner import etl_runner
from pkg.tfrecord_writer.runner import tfrecord_writer_runner
from pkg.modelling.runner import modelling_runner

settings = Settings(
    raw_data_filepath="./data/transactions_train.csv",
    train_data_size=500000,
    test_data_size=100000,
    date_col_name="t_dat",
    candidate_tfrecord_path="./data/tfrecords/candidates/candidates",
    train_data_filepath="./data/train.csv",
    test_data_filepath="./data/test.csv",
    train_data_tfrecord_path="./data/tfrecords/train/train",
    test_data_tfrecord_path="./data/tfrecords/test/test",
    max_tfrecord_rows=100000,
    schema_filepath="./data/schema.pkl"
)

schema = Schema(
    features=[
        Feature(
            "customer_id",
            tf.string,
            FeatureFamily.USER,
            embedding_size=64,
            max_vocab_size=100000,
        ),
        Feature(
            "article_id",
            tf.string,
            FeatureFamily.ITEM,
            embedding_size=64,
            max_vocab_size=100000
        )
    ],
    training_config=TrainingConfig(
        128,
        1000
    ),
    model_config=ModelConfig(
        32,
        [64,64],
        [64,64]
    )
)

# etl_runner(settings, schema)
# tfrecord_writer_runner(settings)
modelling_runner(settings)

