# Adapted from: https://www.tensorflow.org/tutorials/load_data/images
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from .base import Reader, ReaderConfig, UNKNOWN_LABEL
from .._logging import get_logger
from ..custom_types import DataType, DataWithInfo

_logger = get_logger(__name__)

@dataclass(frozen=True)
class NumpyArrayConfig(ReaderConfig):
    path_features: str
    path_labels: str
    height: int
    width: int
    num_channels: int

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


class NumpyArrayReader(Reader):

    @staticmethod
    def read_data(config: NumpyArrayConfig) -> DataWithInfo:

        features = np.load(config.path_features).reshape(-1, config.height, config.width, config.num_channels)
        labels = np.load(config.path_labels)

        data_size = features.shape[0]

        # Load images and labels
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(data_size)

        # Return
        return_value = DataWithInfo(data=dataset, size=data_size, num_labels=len(np.unique(labels)))
        return return_value
