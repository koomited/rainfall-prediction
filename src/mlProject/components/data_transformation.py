
import os 
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        features = list(self.config.all_schema.COLUMNS.keys())
        target = list(self.config.all_schema.TARGET_COLUMN.keys())
        all_variables = features + target
        data = data[all_variables]
        
        data["time"] = pd.to_datetime(data.time)
        
        data_size = len(data)
        test_size = int(self.config.test_size*data_size)
        train_size = data_size-test_size
        train_set = data[:train_size]
        test_set = data[train_size:]

        
        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train_set.shape)
        logger.info(test_set.shape)

