import os
import logging
from datetime import datetime
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
    # Split test ds into tuples of user features and the candidate id
    test_ds = test_ds.map(lambda x: ({f.name: x[f.name] for f in schema.user_features},x[settings.candidate_col_name]))
    candidate_ds = candidate_ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.candidate_tfrecord_path),
        batch_size=100000
    )
    logs = os.path.join(settings.tensorboard_logs_dir,datetime.now().strftime("%Y%m%d-%H%M%S"))
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
        histogram_freq=1,
        profile_batch='20,40'# FIXME: Make dynamic
    )
    file_writer = tf.summary.create_file_writer(logs + "/metrics")
    file_writer.set_as_default()
    model = TwoTowerModel.create_from_schema(schema)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=schema.training_config.optimizer
    )
    for epoch in range(schema.training_config.epochs):
        # Make into 1D for recall calculation
        candidate_embeddings = candidate_ds.map(lambda x: (tf.reshape(x[settings.candidate_col_name],(-1,)), model.item_tower(x)))
        index = BruteForceIndex(max(schema.model_config.ks), model.user_tower, candidate_embeddings)
        metric_calc = IndexRecall(index, schema.model_config.ks)
        for user_features,true_candidates in test_ds:
            metric_calc(user_features, true_candidates)
        metric_calc.log_to_tensorboard(epoch+1)
        model.fit(train_ds, epochs=1, callbacks=[tboard_callback])
        model.save(settings.trained_model_path)
        index.save(settings.index_path)
    metric_calc.log_to_tensorboard(schema.training_config.epochs+1)
    logger.info("--- Modelling Finishing ---")
