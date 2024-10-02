import logging
import pandas as pd
from pkg.etl.transformations import load_dataframe
from pkg.tfrecord_writer.tfrecord_writer import TFRecordWriter
from pkg.utils.settings import Settings
from pkg.schema.schema import Schema

logger = logging.getLogger(__name__)


def tfrecord_writer_runner(settings: Settings) -> None:
    """
    Convert data to TFRecords and save to disk.
    TFRecord Writer steps:
      1. Load train and test data into memory.
      2. Write unique candidate TFrecords to disk.
      3. Write train/test TFRecords to disk.

    Parameters
    ----------
    settings: Settings
      Settings for the run.
    """
    logger.info("--- TFRecord Writing Starting ---")
    # Load CSV data in and Schema
    train = load_dataframe(settings.train_data_filepath, "train")
    test = load_dataframe(settings.test_data_filepath, "test")
    schema = Schema.load_from_filepath(settings.schema_filepath)
    # Write ALL candidates to TFRecords
    logger.info("Creating unique candidates from train/test data")
    logger.info("Candidates will be used to evaluate test data")
    combined = pd.concat([train, test])[
        [f.name for f in schema.candidate_features]
    ]
    logger.info(f"{len(combined)} rows before dropping duplicate candidates")
    # This assumes the same candidate won't have differing features
    combined = combined.drop_duplicates()
    logger.info(f"{len(combined)} rows after dropping duplicate candidates")
    # Write inference candidates to TFRecords
    candidate_writer = TFRecordWriter(schema.candidate_features)
    candidate_writer.write_tfrecords(
        combined, settings.candidate_tfrecord_path, settings.max_tfrecord_rows
    )
    # Write train and test to TFRecords
    writer = TFRecordWriter(schema.features)
    writer.write_tfrecords(
        train, settings.train_data_tfrecord_path, settings.max_tfrecord_rows
    )
    writer.write_tfrecords(
        test, settings.test_data_tfrecord_path, settings.max_tfrecord_rows
    )
    logger.info("--- TFRecord Writing Finishing ---")
