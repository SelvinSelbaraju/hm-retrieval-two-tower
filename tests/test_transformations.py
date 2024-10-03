import pytest
import pandas as pd
from pkg.etl.transformations import date_filter


@pytest.fixture
def dummy_transactions() -> pd.DataFrame:
    data = {
        "date_col": [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
            "2024-05-01",
        ],
        "query_id": ["123", "456", "123", "789", "456"],
        "candidate_id": ["abc", "def", "ghi", "abc", "def"],
    }
    return pd.DataFrame(data)


def test_date_filter(dummy_transactions):
    train = date_filter(
        dummy_transactions, "train", "date_col", ("2024-01-01", "2024-02-01")
    )
    test = date_filter(
        dummy_transactions, "train", "date_col", ("2024-03-01", "2024-04-01")
    )
    assert (train["date_col"].min(), train["date_col"].max()) == (
        "2024-01-01",
        "2024-02-01",
    )
    assert (test["date_col"].min(), test["date_col"].max()) == (
        "2024-03-01",
        "2024-04-01",
    )
