import os
import logging
from typing import List, Dict, Optional
import tensorflow as tf
from pkg.schema.features import Feature

logger = logging.getLogger(__name__)


class TFRecordDatasetFactory:
    """
    Creates TFRecord Datasets given a list of Feature objs.

    Parameters
    ----------
    features: List[Feature]
        List of Feature objs containing tf dtypes.
    """

    def __init__(self, features: List[Feature]):
        self.features = features
        self.feature_description = self._create_feature_description()

    def _create_feature_description(self) -> Dict[str, tf.io.FixedLenFeature]:
        """
        Create the Feature Description to parse raw TFRecords.
        The shape is (1,), so becomes (1,1) when batched.

        Returns
        -------
        feature_description: Dict[str, tf.io.FixedLenFeature]
            A dict describing each feature's shape and type when parsed.
        """
        return {
            feature.name: tf.io.FixedLenFeature([1], feature.dtype)
            for feature in self.features
        }

    def _parse_function(
        self, example_proto: tf.train.Example
    ) -> Dict[str, tf.Tensor]:
        """
        Parse a raw TFRecord proto from a TFRecord file.

        Parameters
        ----------
        example_proto: tf.train.Example
          A single raw TFRecord example.

        Returns
        -------
        parsed_example: Dict[str, tf.Tensor]
            A dict of features to tensors.
        """
        return tf.io.parse_single_example(
            example_proto, self.feature_description
        )

    def create_tfrecord_dataset(
        self,
        file_dir: str,
        batch_size: Optional[int] = None,
        shuffle_size: Optional[int] = None,
    ) -> tf.data.TFRecordDataset:
        """
        Create a TFRecord Dataset from a directory of TFRecords.
        Returns a TFRecordDataset object.

        Parameters
        ----------
        file_dir: str
            Directory where TFRecords are stored.
        batch_size: Optional[int]
            Yield batches of these size if provided.
        shuffle_size: Optional[int]
            Shuffle buffer of this size if provided.

        Returns
        -------
        ds: tf.data.TFRecordDataset
            TF Dataset which lazily loads data.
        """
        filenames = [
            os.path.join(file_dir, file)
            for file in os.listdir(file_dir)
            if file.endswith(".tfrecord")
        ]
        ds = tf.data.TFRecordDataset(filenames)
        ds = ds.map(lambda x: self._parse_function(x))
        if shuffle_size:
            logger.info(
                f"Shuffling dataset using shuffle size: {shuffle_size}"
            )
            ds = ds.shuffle(shuffle_size)
        if batch_size:
            logger.info(f"Batching data using batch_size: {batch_size}")
            ds = ds.batch(batch_size)
        return ds
