from typing import Dict, Any
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class OptimizerFactory:
    # Legacy optimizers are currently much faster
    _supported_optimizers = {
        "adam": tf.keras.optimizers.legacy.Adam,
        "adagrad": tf.keras.optimizers.legacy.Adagrad,
    }

    _required_kwargs = ["learning_rate"]

    @classmethod
    def get_optimizer(
        cls, optimizer_name: str, optimizer_kwargs: Dict[str, Any]
    ) -> tf.keras.optimizers.legacy.Optimizer:
        if optimizer_name not in cls._supported_optimizers:
            raise ValueError(
                "name must be one of "
                f"{list(cls._supported_optimizers.keys())}, "
                f"got {optimizer_name}"
            )
        for kwarg in cls._required_kwargs:
            if kwarg not in optimizer_kwargs:
                raise ValueError(
                    f"kwarg {kwarg} not found in kwargs: {optimizer_kwargs}"
                )
        logger.info(
            f"Creating {optimizer_name} obj with kwargs: {optimizer_kwargs}"
        )
        return cls._supported_optimizers[optimizer_name](**optimizer_kwargs)
