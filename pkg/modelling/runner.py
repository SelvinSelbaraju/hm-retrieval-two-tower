import os
import logging
import tensorflow as tf
from pkg.modelling.tfrecord_dataset import TFRecordDatasetFactory
from pkg.utils.settings import Settings
from pkg.schema.schema import Schema
from pkg.modelling.models.two_tower_model import TwoTowerModel

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
    candidate_ds = candidate_ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.candidate_tfrecord_path),
        1
    )
    # model = TwoTowerModel.create_from_schema(schema)
    # model.compile(
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #     optimizer="adam"
    # )
    # model.fit(train_ds, epochs=1)
    logger.info("--- Modelling Finishing ---")
