from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Settings:
    """
    Settings for running the code e2e.

    Parameters
    ----------
    raw_data_filepath: str
        Filepath to the raw data.
    articles_data_filepath: str
        Filepath to the articles metadata.
    customers_data_filepath: str
        Filepath to the customers metadata.
    train_data_range: Tuple[str, str]
        Start and end date for training data
        in the format YYYY-MM-DD.
    test_data_range: Tuple[str, str]
        Start and end date for test data
        in the format YYYY-MM-DD.
    baseline_model_date_range: Tuple[str, str]
        Start and end date for baseline model
        in the format YYYY-MM-DD.
    date_col_name: str
        Name of the col with the date/time.
        Used to prevent leakage in train/test.
    candidate_col_name: str
        Name of the col with the candidate identifier.
    candidate_tfrecord_path: str
        Filepath to store TFRecords of candidates.
    train_data_filepath: str
        Filepath to save and read train CSV data.
    test_data_filepath: str
        Filepath to save and read test CSV data.
    train_data_tfrecord_path: str
        Filepath to save and read train TFRecords.
    test_data_tfrecord_path: str
        Filepath to save and read test TFRecords.
    schema_filepath: str
        Filepath to save and read the Schema obj.
    trained_model_path: str
        Path to save the trained Two-Tower Model.
    index_path: str
        Path to save an index with candidate embeddings.
        Index contains logic to fetch the top K given a query.
    tensorboard_logs_dir: str = "./logs"
        Path to save Tensorboard logs.
    max_tfrecord_rows: Optional[int]
        Max number of rows in a single TFRecord file.
    """

    raw_data_filepath: str
    articles_data_filepath: str
    customers_data_filepath: str
    train_data_range: Tuple[str, str]
    test_data_range: Tuple[str, str]
    baseline_model_date_range: Tuple[str, str]
    date_col_name: str
    candidate_col_name: str
    candidate_tfrecord_path: str
    train_data_filepath: str
    test_data_filepath: str
    train_data_tfrecord_path: str
    test_data_tfrecord_path: str
    schema_filepath: str
    trained_model_path: str
    index_path: str
    baseline_index_path: str
    tensorboard_logs_dir: str = "./logs"
    max_tfrecord_rows: Optional[int] = None
