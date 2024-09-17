from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    """
    Settings for running the code e2e

    Parameters
    ----------
    raw_data_filepath: str
        Filepath to the raw data
    train_data_size: int
        Number of rows in train data
    test_data_size: int
        Approx number of rows in test data
        Approx because train/test overlap days removed
    date_col_name: str
        Name of the col with the date/time
        Used to prevent leakage in train/test
    candidate_tfrecord_path: str
        Filepath to store TFRecords of candidates
    train_data_filepath: str
        Filepath to save and read train CSV data
    test_data_filepath: str
        Filepath to save and read test CSV data
    train_data_tfrecord_path: str
        Filepath to save and read train TFRecords
    test_data_tfrecord_path: str
        Filepath to save and read test TFRecords
    schema_filepath: str
        Filepath to save and read the Schema obj
    max_tfrecord_rows: Optional[int]
        Max number of rows in a single TFRecord file
    """
    raw_data_filepath: str
    train_data_size: int
    test_data_size: int
    date_col_name: str
    candidate_tfrecord_path: str
    train_data_filepath: str
    test_data_filepath: str
    train_data_tfrecord_path: str
    test_data_tfrecord_path: str
    schema_filepath: str
    max_tfrecord_rows: Optional[int] = None
