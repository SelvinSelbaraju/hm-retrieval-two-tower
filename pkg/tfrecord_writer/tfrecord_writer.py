import os
import logging
from typing import List, Union, Optional
import pandas as pd
import tensorflow as tf
from pkg.schema.features import Feature

logger = logging.getLogger(__name__)


class TFRecordWriter:
    """
    Obj for writing TFRecords to disk.

    Parameters
    ----------
    features: List[Feature]
        List of feature objects.
        Usually come from a Schema obj.
    """

    def __init__(self, features: List[Feature]):
        self.features = features

    def _parse_feature(
        self, feature_val: Union[str, float, int], dtype: tf.dtypes.DType
    ) -> tf.train.Feature:
        """
        Given a feature val, return a tf.train.Feature of the correct dtype.

        Parameters
        ----------
        feature_val: Union[str, float, int]
            The piece of data to convert.
        dtype: tf.dtypes.DType
            The Tensorflow dtype to use.
            Must be one of the dtypes set in the Feature obj.
        """
        if dtype == tf.string:
            val = str(feature_val).encode()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))
        elif dtype == tf.float32:
            return tf.train.Feature(float_list=tf.train.FloatList(value=[val]))
        else:
            raise TypeError(
                f"Invalid dtype {dtype} provided, "
                f"must be one of {Feature.VALID_DTYPES}"
            )

    def _get_features_from_row(self, row: tuple) -> bytes:
        """
        Given a NamedTuple row from a Pandas DataFrame,
        return a TFRecord Example serialized to Bytes.

        Parameters
        ----------
        row: tuple
           NamedTuple row from pd.DataFrame.itertuples().
        """
        features = {
            feature.name: self._parse_feature(
                getattr(row, feature.name), feature.dtype
            )
            for feature in self.features
        }
        return tf.train.Example(
            features=tf.train.Features(feature=features)
        ).SerializeToString()

    def write_tfrecords(
        self,
        df: pd.DataFrame,
        filepath: str,
        max_file_size: Optional[int] = None,
    ) -> None:
        """
        Write data to TFRecords based on features in the schema.
        Saves TFRecord files in the filepath.
        Number of examples in a file can be limited into partitions.
        The suffix contains the partition number.

        Parameters
        ----------
        df: pd.DataFrame
            Data to write to TFRecords.
        filepath: str
            Filepath to store TFRecords.
            Must not have .tfrecord suffix, removed if so.
        """
        logger.info("Removing .tfrecord suffix if present")
        filepath = filepath.replace(".tfrecord", "")
        # Create any directories
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Limit the file size if provided
        if max_file_size:
            num_files = len(df) // max_file_size
            # Need to add 1 if the max file size is not a perfect divisor
            if len(df) % max_file_size:
                num_files += 1
            logger.info(
                f"Num files: {num_files} with max rows {max_file_size}"
            )
        else:
            num_files = 1
            max_file_size = len(df)
            logger.info("Writing all rows to single file")
        for file_num in range(num_files):
            start = file_num * max_file_size
            end = (file_num + 1) * max_file_size
            rows = df.iloc[start:end]
            tfrecord_filename = f"{filepath}_{file_num}.tfrecord"
            logger.info(f"Writing to TFRecord file: {tfrecord_filename}")
            with tf.io.TFRecordWriter(tfrecord_filename) as writer:
                for row in rows.itertuples():
                    example = self._get_features_from_row(row)
                    writer.write(example)
