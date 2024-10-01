import tensorflow as tf
from pkg.schema.model_config import ModelConfig
from pkg.schema.schema import Schema
from pkg.schema.training_config import TrainingConfig
from pkg.schema.features import Feature, FeatureFamily
from pkg.utils.settings import Settings
from pkg.etl.runner import etl_runner, build_schema_runner
from pkg.tfrecord_writer.runner import tfrecord_writer_runner
from pkg.modelling.runner import modelling_runner, baseline_modelling_runner

settings = Settings(
    raw_data_filepath="./data/transactions_train.csv",
    articles_data_filepath="./data/articles.csv",
    customers_data_filepath="./data/customers.csv",
    train_data_range=("2019-09-20", "2020-08-20"),
    test_data_range=("2020-08-21", "2020-09-21"),
    baseline_model_date_range=("2019-09-20", "2020-08-20"),
    date_col_name="t_dat",
    candidate_col_name="article_id",
    candidate_tfrecord_path="./data/tfrecords/candidates/candidates",
    train_data_filepath="./data/train.csv",
    test_data_filepath="./data/test.csv",
    train_data_tfrecord_path="./data/tfrecords/train/train",
    test_data_tfrecord_path="./data/tfrecords/test/test",
    max_tfrecord_rows=100000,
    schema_filepath="./data/schema.pkl",
    trained_model_path="./trained_models/model/",
    index_path="./trained_models/candidate_index",
    baseline_index_path="./trained_models/baseline_index",
)

schema = Schema(
    features=[
        # Customer features
        Feature(
            "customer_id",
            tf.string,
            FeatureFamily.QUERY,
            embedding_size=128,
        ),
        Feature(
            "fashion_news_frequency",
            tf.string,
            FeatureFamily.QUERY,
            embedding_size=2,
        ),
        Feature(
            "postal_code", tf.string, FeatureFamily.QUERY, embedding_size=128
        ),
        # Article features
        Feature(
            "article_id",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=128,
        ),
        Feature(
            "product_type_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=16,
        ),
        Feature(
            "product_type_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=4,
        ),
        Feature(
            "colour_group_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=8,
        ),
        Feature(
            "department_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=32,
        ),
        Feature(
            "index_name", tf.string, FeatureFamily.CANDIDATE, embedding_size=4
        ),
        Feature(
            "section_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=16,
        ),
        Feature(
            "garment_group_name",
            tf.string,
            FeatureFamily.CANDIDATE,
            embedding_size=4,
        ),
    ],
    training_config=TrainingConfig(
        train_batch_size=512,
        test_batch_size=2048,
        optimizer_name="adagrad",
        optimizer_kwargs={"learning_rate": 0.05},
        shuffle_size=100000,
        epochs=5,
    ),
    model_config=ModelConfig(
        joint_embedding_size=128,
        ks=[10, 100, 1000],
        query_tower_units=[256],
        candidate_tower_units=[256],
    ),
)

etl_runner(settings)
build_schema_runner(settings, schema)
tfrecord_writer_runner(settings)
modelling_runner(settings)
baseline_modelling_runner(settings)
