import logging
import os
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def date_filter(
    df: pd.DataFrame, df_name: str, date_col: str, date_range: Tuple[str, str]
) -> pd.DataFrame:
    """
    Filter a df into a specified date range.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to split.
    df_name: pd.DataFrame
        The name of the dataframe.
        Used for logging.
    date_col: str
        The name of the date col in df.
    date_range: Tuple[str, str]
        The start and end date
        in the format YYYY-MM-DD.

    Returns
    -------
    filtered: pd.DataFrame
        The df filtered.
    """
    logger.info(
        f"Creating df {df_name} from: "
        f"{date_range[0]} "
        f"to: {date_range[1]}"
    )
    filtered = df[
        (df[date_col] >= date_range[0]) & (df[date_col] <= date_range[1])
    ]
    return filtered


def load_dataframe(path: str, df_name: str) -> pd.DataFrame:
    """
    Load the datframe from the specified CSV path.
    Simple wrapper to include logging.

    Parameters
    ----------
    path: str
        The path where the raw CSV is stored.
    df_name: str
        The name of the dataframe.
        Used for logging.

    Returns
    -------
    df: pd.DataFrame
        The loaded in dataframe.
    """
    logger.info(f"Loading df {df_name} from {path}")
    df = pd.read_csv(path)
    return df


def save_dataframe(
    df: pd.DataFrame, df_name: str, date_col: str, path: str
) -> None:
    """
    Save the df at the desired location.
    Create the directory if does not exist.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe to save.
    df_name: str
        The name of the dataframe.
        Used for logging.
    date_col: str
        The column containing the date.
        Used for logging.
    path: str
        The path to save the dataframe.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(
        f"Saving {len(df)} rows from df {df_name} to: "
        f"{path}. Date range: "
        f"{df[date_col].min()} to: "
        f"{df[date_col].max()}"
    )
    df.to_csv(path, index=False)
    logger.info(f"Finished saving df {df_name}")
