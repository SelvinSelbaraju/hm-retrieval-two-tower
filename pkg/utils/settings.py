from dataclasses import dataclass

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
    train_data_filepath: str
        Filepath to save and read train data
    test_data_filepath: str
        Filepath to save and read test data
    schema_filepath: str
        Filepath to save and read the Schema obj
    """
    raw_data_filepath: str
    train_data_size: int
    test_data_size: int
    date_col_name: str
    train_data_filepath: str
    test_data_filepath: str
    schema_filepath: str
