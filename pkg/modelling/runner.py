import os
import logging
import tensorflow as tf
from pkg.modelling.tfrecord_dataset import TFRecordDatasetFactory
from pkg.utils.settings import Settings
from pkg.schema.schema import Schema
from pkg.modelling.models.two_tower_model import TwoTowerModel
from pkg.modelling.indices.brute_force import BruteForceIndex
from pkg.modelling.metrics.index_recall import IndexRecall

logger = logging.getLogger(__name__)

"""
Modelling steps:
  1. Create TFRecord Datasets for train/test data
  2. Create model class given Schema args
  3. Train model and evaluate each epoch
"""
def modelling_runner(settings: Settings):
    logger.info("--- Modelling Starting ---")
    schema = Schema.load_from_filepath(settings.schema_filepath)
    ds_factory = TFRecordDatasetFactory(schema.features)
    candidate_ds_factory = TFRecordDatasetFactory(schema.item_features)
    train_ds = ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.train_data_tfrecord_path),
        schema.training_config.train_batch_size,
        schema.training_config.shuffle_size,
    )
    test_ds = ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.test_data_tfrecord_path),
        batch_size=schema.training_config.test_batch_size,
    )
    candidate_ds = candidate_ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.candidate_tfrecord_path),
    )
    model = TwoTowerModel.create_from_schema(schema)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.05)
    )
    for i in range(schema.training_config.epochs):
        model.save(settings.trained_model_path)
        index = BruteForceIndex(1000, model.user_tower)
        candidate_embeddings = candidate_ds.map(lambda x: (x[settings.candidate_col_name], model.item_tower(x)))
        # DEBUGGING
        for articles,embeddings in candidate_embeddings.take(1):
            logger.info(f"id: {articles[0]}, embedding: {embeddings[0]}")
        index.index(candidate_embeddings)
        metric_calc = IndexRecall(index)
        for batch in test_ds:
            metric_calc(batch, batch[settings.candidate_col_name])
        logger.info(f"Start of epoch {i} recall:{metric_calc.metric.numpy()}")
        model.fit(train_ds, epochs=1)
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam"
        )
    logger.info(f"Recall@k: {metric_calc.metric.numpy()}")
    logger.info("--- Modelling Finishing ---")
