from abc import ABC, abstractmethod
from typing import Dict
import logging
import os
import tensorflow as tf

logger = logging.getLogger(__name__)


class AbstractKerasModel(ABC, tf.keras.Model):
    """
    Abstract class with functions for keras models.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_input_signature(self) -> Dict[str, tf.TensorSpec]:
        """
        Given instance attributes, return the input signature.
        Different for every model.

        Returns
        -------
        input_signature: Dict[str, tf.TensorSpec]
            A dict mapping features to TensorSpec objs.
        """

    def set_input_signature(
        self, input_signature: Dict[str, tf.TensorSpec]
    ) -> None:
        """
        Set what the inputs to the call function should be.
        This is needed so the call method is saved.

        Parameters
        ----------
        input_signature: Dict[str, tf.TensorSpec]
            A map of inputs to tf.Tensorspec objs.
        """
        self.__call__ = tf.function(
            self.call, input_signature=[{**input_signature}]
        )

    @staticmethod
    def _get_default_tensor(dtype: tf.dtypes.DType) -> tf.Tensor:
        """
        For each dtype, say what to use as the default tensor.

        Parameters
        ----------
        dtype: tf.dtypes.DType
            One of tf.string or tf.float32.
            Used for passing inputs through the model.
            Allows the call method of a model to be traced.

        Returns
        -------
        tensor: tf.Tensor
            A tensor containing a simple default for the dtype.
        """
        if dtype == tf.string:
            return tf.constant([b"a"], shape=(1, 1), dtype=tf.string)
        elif dtype == tf.float32:
            return tf.constant([0.0], shape=(1, 1), dtype=tf.float32)
        else:
            raise TypeError(f"Invalid dtype {dtype}")

    def get_default_inputs(
        self, input_signature: Dict[str, tf.TensorSpec]
    ) -> Dict[str, tf.Tensor]:
        """
        Create default inputs to build the model.

        Parameters
        ----------
        input_signature: Dict[str, tf.TensorSpec]
            A map of inputs to tf.Tensorspec objs.

        Returns
        -------
        default_inputs: Dict[str, tf.Tensor]
            Dict[str, tf.Tensor] of inputs to tensors.
        """
        inputs = {}
        for f, spec in input_signature.items():
            inputs[f] = self._get_default_tensor(spec.dtype)
        return inputs

    @abstractmethod
    def call(
        self, x: Dict[str, tf.Tensor], training: bool = True
    ) -> tf.Tensor:
        """
        Pass data through the model.

        Parameters
        ----------
        x: Dict[str, tf.Tensor]
            Dict of features to tensors.

        Returns
        -------
        results: tf.Tensor
            Output from the model.
        """

    def initialise_model(self) -> None:
        """
        Run the steps required to set up the model:
        1. Set the input signature for call.
        2. Pass dummy inputs to the model.
        """
        input_signature = self.get_input_signature()
        self.set_input_signature(input_signature)
        default_inputs = self.get_default_inputs(input_signature)
        self.__call__(default_inputs)

    def save(self, model_path: str) -> None:
        """
        Save the model at the path.

        Parameters
        ----------
        model_path: str
            The path to save the model at
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logging.info(f"Saving model at path: {model_path}")
        tf.saved_model.save(self, model_path)
