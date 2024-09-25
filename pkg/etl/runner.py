import logging
import pandas as pd
from pkg.schema.schema import Schema
from pkg.utils.settings import Settings

logger = logging.getLogger(__name__)

"""
ETL steps:
  1. Load data into memory
  2. Create train/test data from settings
  3. Save to disk
"""


def etl_runner(settings: Settings) -> None:
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
    logger.info(f"Loading data from {settings.raw_data_filepath}")
    data = pd.read_csv(settings.raw_data_filepath)
    logger.info(
        "Creating train data from: "
        f"{settings.train_data_range[0]} "
        f"to: {settings.train_data_range[1]}"
    )
    train = data[
        (data[settings.date_col_name] >= settings.train_data_range[0])
        & (data[settings.date_col_name] <= settings.train_data_range[1])
    ]
    logger.info(
        "Creating test data from: "
        f"{settings.test_data_range[0]} "
        f"to: {settings.test_data_range[1]}"
    )
    test = data[
        (data[settings.date_col_name] >= settings.test_data_range[0])
        & (data[settings.date_col_name] <= settings.test_data_range[1])
    ]
    # Save the data
    logger.info(
        f"Saving {len(train)} rows train data to: "
        f"{settings.train_data_filepath}. Date range: "
        f"{train[settings.date_col_name].min()} to: "
        f"{train[settings.date_col_name].max()}"
    )
    train.to_csv(settings.train_data_filepath, index=False)
    logger.info(
        f"Saving {len(test)} rows test data to: "
        f"{settings.test_data_filepath}. Date range: "
        f"{test[settings.date_col_name].min()} to: "
        f"{test[settings.date_col_name].max()}"
    )
    test.to_csv(settings.test_data_filepath, index=False)
    logger.info("--- ETL Finished! ---")


def build_schema_runner(settings: Settings, schema: Schema) -> None:
    """
    Build the schema from the training data

    Parameters
    ----------
    settings: Settings
      Settings for the run
    schema: Schema
      Schema containing features
    """
    logger.info("--- Build Schema Starting ---")
    logger.info("Building schema from training data")
    train = pd.read_csv(settings.train_data_filepath)
    schema.build_features_from_dataframe(train)
    logger.info("Calculating candidate probs from training data")
    probs = train[settings.candidate_col_name].value_counts() / len(train)
    lookup_dict = {
        str(probs.index[i]): probs.iloc[i] for i in range(len(probs))
    }
    logger.info(
        f"Finished creating lookup dict with {len(lookup_dict)} candidates"
    )
    schema.set_candidate_prob_lookup(lookup_dict)
    schema.save(settings.schema_filepath)
    logger.info("--- Build Schema Finished! ---")
