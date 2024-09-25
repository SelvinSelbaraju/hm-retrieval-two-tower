import logging
import pandas as pd
from pkg.schema.schema import Schema
from pkg.utils.settings import Settings
from pkg.etl.transformations import date_filter, load_dataframe, save_dataframe

logger = logging.getLogger(__name__)


def etl_runner(settings: Settings) -> None:
    """
    Given the settings, output train/test data
    ETL steps:
      1. Load data into memory
      2. Create train/test data from settings
      3. Save to disk

    Parameters
    ----------
    settings: Settings
      Settings for the run
    """
    logger.info("--- ETL Starting ---")
    # Load the data
    transactions = load_dataframe(
        settings.raw_data_filepath, "raw_transactions"
    )
    # Create train/test
    train = date_filter(
        transactions,
        "train",
        settings.date_col_name,
        settings.train_data_range,
    )
    test = date_filter(
        transactions, "test", settings.date_col_name, settings.test_data_range
    )
    # Save the data
    save_dataframe(
        train, "train", settings.date_col_name, settings.train_data_filepath
    )
    save_dataframe(
        test, "test", settings.date_col_name, settings.test_data_filepath
    )
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
