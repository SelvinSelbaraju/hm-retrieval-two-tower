import logging
import pandas as pd
from pkg.tfrecord_writer.tfrecord_writer import TFRecordWriter
from pkg.utils.settings import Settings
from pkg.schema.schema import Schema

logger = logging.getLogger(__name__)

"""
TFRecord Writer Steps
    1. Load CSV in
    2. Write to TFRecords
"""
def tfrecord_writer_runner(settings: Settings) -> None:
    logger.info("--- TFRecord Writing Starting ---")
    # Load CSV data in and Schema
    train = pd.read_csv(settings.train_data_filepath)
    test = pd.read_csv(settings.test_data_filepath)
    schema = Schema.load_from_filepath(settings.schema_filepath)
    # Write ALL candidates to TFRecords
    combined = pd.concat([train, test])[[f.name for f in schema.item_features]]
    logger.info(f"{len(combined)} rows before dropping duplicate candidates")
    combined = combined.drop_duplicates()
    logger.info(f"{len(combined)} rows after dropping duplicate candidates")
    candidate_writer = TFRecordWriter(schema.item_features)
    candidate_writer.write_tfrecords(combined, settings.candidate_tfrecord_path, settings.max_tfrecord_rows)
    # Write train and test to TFRecords
    writer = TFRecordWriter(schema.features)
    writer.write_tfrecords(train, settings.train_data_tfrecord_path, settings.max_tfrecord_rows)
    writer.write_tfrecords(test, settings.test_data_tfrecord_path, settings.max_tfrecord_rows)
    logger.info("--- TFRecord Writing Finishing ---")


