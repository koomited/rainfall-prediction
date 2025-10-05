import os
from pathlib import Path

import pandas as pd

from mlProject import logger
from mlProject.utils.common import get_size
from mlProject.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            data = pd.read_csv(self.config.source_URL)
            data.to_csv(self.config.local_data_file, index=False)
            logger.info(f"data download with following info: shape:{data.shape}")
        else:
            logger.info(
                f"File already exists with size: {get_size(Path(self.config.local_data_file))} bytes"
            )
