from pkg.utils.settings import Settings
from pkg.etl.runner import etl_runner

settings = Settings(
    raw_data_filepath="./data/transactions_train.csv",
    train_data_size=500000,
    test_data_size=100000,
    date_col_name="t_dat",
    train_data_filepath="./data/train.csv",
    test_data_filepath="./data/test.csv",
    schema_filepath="FIX ME!"
)

etl_runner(settings)


