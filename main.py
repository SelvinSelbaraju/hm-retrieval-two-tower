import tensorflow as tf
from pkg.schema.schema import Schema
from pkg.schema.features import Feature
from pkg.utils.settings import Settings
from pkg.etl.runner import etl_runner
from pkg.tfrecord_writer.runner import tfrecord_writer_runner

settings = Settings(
    raw_data_filepath="./data/transactions_train.csv",
    train_data_size=500000,
    test_data_size=100000,
    date_col_name="t_dat",
    train_data_filepath="./data/train.csv",
    test_data_filepath="./data/test.csv",
    train_data_tfrecord_path="./data/tfrecords/train",
    test_data_tfrecord_path="./data/tfrecords/test",
    max_tfrecord_rows=100000,
    schema_filepath="./data/schema.pkl"
)

schema = Schema(
    features=[
        Feature(
            "customer_id",
            tf.string,
            max_vocab_size=100000,
        ),
        Feature(
            "article_id",
            tf.string,
            max_vocab_size=100000
        )
    ]
)

# etl_runner(settings, schema)
tfrecord_writer_runner(settings)


