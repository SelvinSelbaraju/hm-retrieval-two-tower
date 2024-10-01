import os
import logging
from datetime import datetime
import tensorflow as tf
from pkg.etl.transformations import load_dataframe, date_filter
from pkg.modelling.tfrecord_dataset import TFRecordDatasetFactory
from pkg.utils.settings import Settings
from pkg.schema.schema import Schema
from pkg.modelling.optimizer_factory import OptimizerFactory
from pkg.modelling.models.two_tower_model import TwoTowerModel
from pkg.modelling.indices.brute_force import BruteForceIndex
from pkg.modelling.indices.static_index import StaticIndex
from pkg.modelling.metrics.index_recall import IndexRecall

logger = logging.getLogger(__name__)


def modelling_runner(settings: Settings):
    """
    Train a Two-Tower Model and evaluate it
    Modelling steps:
        1. Create TFRecord Datasets for train/test/candidates data
        2. Create Tensorboard callback for logs
        3. Create model class given Schema args
        4. Train model and evaluate each epoch

    Parameters
    ----------
    settings: Settings
      Settings for the run
    """
    logger.info("--- Modelling Starting ---")
    schema = Schema.load_from_filepath(settings.schema_filepath)
    ds_factory = TFRecordDatasetFactory(schema.features)
    candidate_ds_factory = TFRecordDatasetFactory(schema.candidate_features)
    train_ds = ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.train_data_tfrecord_path),
        schema.training_config.train_batch_size,
        schema.training_config.shuffle_size,
    )
    test_ds = ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.test_data_tfrecord_path),
        batch_size=schema.training_config.test_batch_size,
    )
    # Split test ds into tuples of query features and the candidate id
    # This structure is used for evaluation
    test_ds = test_ds.map(
        lambda x: (
            {f.name: x[f.name] for f in schema.query_features},
            x[settings.candidate_col_name],
        )
    )
    candidate_ds = candidate_ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.candidate_tfrecord_path),
        batch_size=schema.training_config.candidate_batch_size,
    )
    # Create Tensorboard Logs Callback
    logs = os.path.join(
        settings.tensorboard_logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logs,
        histogram_freq=1,
        profile_batch="20,40",  # FIXME: Make dynamic
    )
    file_writer = tf.summary.create_file_writer(logs + "/metrics")
    file_writer.set_as_default()
    # Create the model class from the schema
    model = TwoTowerModel.create_from_schema(schema)
    os.environ["TF_USE_LEGACY_KERAS"] = (
        "True"  # required for legacy optimizers
    )
    optimizer = OptimizerFactory.get_optimizer(
        schema.training_config.optimizer_name,
        schema.training_config.optimizer_kwargs,
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM
        ),
        optimizer=optimizer,
    )
    # Train and evaluate each epoch
    for epoch in range(schema.training_config.epochs):
        # Make into 1D for recall calculation
        candidate_embeddings = candidate_ds.map(
            lambda x: (
                tf.reshape(x[settings.candidate_col_name], (-1,)),
                model.candidate_tower(x),
            )
        )
        index = BruteForceIndex(
            max(schema.model_config.ks),
            model.query_tower,
            candidate_embeddings,
        )
        metric_calc = IndexRecall(index, schema.model_config.ks)
        for query_features, true_candidates in test_ds:
            metric_calc(query_features, true_candidates)
        metric_calc.log_metric(epoch + 1)
        model.fit(train_ds, epochs=1, callbacks=[tboard_callback])
        model.save(settings.trained_model_path)
        index.save(settings.index_path)
    metric_calc.log_metric(schema.training_config.epochs + 1)
    logger.info("--- Modelling Finishing ---")


def baseline_modelling_runner(settings: Settings):
    """
    Create and evaluate a popularity-based heuristic model
    Baseline Modelling Runner Steps:
    1. Load the training data in
    2. Filter the data to a desired range
    3. Build an index from the training data
    4. Evaluate it
    """
    logger.info("--- Baseline Modelling Starting ---")
    df = load_dataframe(settings.raw_data_filepath, "raw_transactions")
    schema = Schema.load_from_filepath(settings.schema_filepath)
    candidates = date_filter(
        df,
        "raw_transactions",
        settings.date_col_name,
        settings.baseline_model_date_range,
    )[settings.candidate_col_name]
    logger.info(
        f"Building Static Popularity Index using {len(candidates)} candidates"
    )
    ds_factory = TFRecordDatasetFactory(schema.features)
    test_ds = ds_factory.create_tfrecord_dataset(
        os.path.dirname(settings.test_data_tfrecord_path),
        batch_size=schema.training_config.test_batch_size,
    )
    # Split test ds into tuples of query features and the candidate id
    test_ds = test_ds.map(
        lambda x: (
            {f.name: x[f.name] for f in schema.query_features},
            x[settings.candidate_col_name],
        )
    )
    index = StaticIndex.build_popularity_index_from_series_schema(
        schema, candidates
    )
    metric_calc = IndexRecall(index, schema.model_config.ks)
    for query_features, true_candidates in test_ds:
        metric_calc(query_features, true_candidates)
    metric_calc.log_metric(None, to_tensorboard=False)
    index.save(settings.baseline_index_path)
    logger.info("--- Baseline Modelling Finishing ---")
