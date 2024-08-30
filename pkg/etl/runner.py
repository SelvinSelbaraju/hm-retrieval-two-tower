import logging
import pandas as pd
from pkg.schema.schema import Schema
from pkg.utils.settings import Settings

logger = logging.getLogger(__name__)

"""
ETL steps:
  1. Create a dataloader from the raw csv, so don't load all data into memory
  2. Create train/test data, ensuring no leakage
"""

def etl_runner(settings: Settings, schema: Schema) -> None:
    """
    Given the settings, output train/test data

    Parameters
    ----------
    settings: Settings
      Settings for the run
    schema: Schema
      Schema containing features
    """
    logger.info("--- ETL Starting ---")
    logger.info(f"Creating iterator from {settings.raw_data_filepath}")
    data_loader = pd.read_csv(settings.raw_data_filepath, chunksize=100000)
    train = data_loader.get_chunk(settings.train_data_size)
    test = data_loader.get_chunk(settings.test_data_size)
    # Prevent overlapping days so no leakage
    max_train_day = train[settings.date_col_name].max()
    logger.info(f"Train data ends on {max_train_day}, removing rows from test...")
    test = test[test[settings.date_col_name] > max_train_day]
    # Save the data
    logger.info(f"Saving train data to {settings.train_data_filepath} with date range {train[settings.date_col_name].min()} to {train[settings.date_col_name].max()}")
    train.to_csv(settings.train_data_filepath, index=False)
    logger.info(f"Saving test data to {settings.test_data_filepath} with date range {test[settings.date_col_name].min()} to {test[settings.date_col_name].max()}")
    test.to_csv(settings.test_data_filepath, index=False)
    # Build and save the schema
    logger.info("Building schema from training data")
    schema.build_features_from_dataframe(train)
    schema.save(settings.schema_filepath)
    logger.info("--- ETL Finished! ---")
