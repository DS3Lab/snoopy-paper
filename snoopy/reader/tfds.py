from dataclasses import dataclass
from typing import Union, Tuple

import tensorflow_datasets as tfds

from .base import Reader, ReaderConfig
from .._cache import get_tfds_cache_dir
from .._logging import get_logger
from ..custom_types import DataType, DataWithInfo

_logger = get_logger(__name__)


@dataclass(frozen=True)
class TFDSImageConfig(ReaderConfig):
    dataset_name: str
    split: tfds.Split
    keys: Tuple[str, str] = None

    @property
    def data_type(self) -> DataType:
        return DataType.IMAGE


@dataclass(frozen=True)
class TFDSTextConfig(ReaderConfig):
    dataset_name: str
    split: tfds.Split
    keys: Tuple[str, str] = None

    @property
    def data_type(self) -> DataType:
        return DataType.TEXT


class TFDSReader(Reader):
    @staticmethod
    def read_data(config: Union[TFDSImageConfig, TFDSTextConfig]) -> DataWithInfo:
        info: tfds.core.DatasetInfo
        data, info = tfds.load(config.dataset_name, split=config.split, data_dir=get_tfds_cache_dir(),
                               shuffle_files=True, with_info=True, as_supervised=config.keys is None)

        if config.keys is not None:
            data = data.map(lambda x: (x[config.keys[0]], x[config.keys[1]]))

        data_size = info.splits[config.split].num_examples
        num_labels = info.features["label"].num_classes
        return_value = DataWithInfo(data=data.shuffle(data_size), size=data_size, num_labels=num_labels)
        _logger.debug(f"Loaded text dataset {info.name} (split: {config.split}) with {data_size} points")
        return return_value
