
import os 
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.config.configuration import DataTransformationConfig
import numpy as np
from copy import deepcopy as dc

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        
    def prepare_dataframe_for_lstm(self, df, 
                                   n_steps=7):
        df = dc(df)
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace = True)
        variables = df.columns
        for variable in variables:
            for i in range(1, n_steps +1):
                df[f"{variable}(t-{i})"]=df[variable].shift(i)
        df.dropna(inplace=True)
        return df
        
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        features = list(self.config.all_schema.COLUMNS.keys())
        target = list(self.config.all_schema.TARGET_COLUMN.keys())
        all_variables = features + target
        data = data[all_variables]

        
        data_size = len(data)
        test_size = int(self.config.test_size*data_size)
        train_size = data_size-test_size
        train_set = data.iloc[:train_size].copy()
        test_set = data.iloc[train_size:].copy()


        train_set.loc[:, "is_clog"] = train_set[features].isna().all(axis=1)
        test_set.loc[:, "is_clog"] = test_set[features].isna().all(axis=1)
        
        train_set.loc[train_set["is_clog"], target[0]] = np.nan
        test_set.loc[test_set["is_clog"], target[0]] = np.nan
        
        features_no_time_clog = [f for f in features if f not in ["time", "is_clog"]]
        train_set.loc[:, features_no_time_clog + target] = train_set[features_no_time_clog + target].interpolate(method="spline", order=3) 
        test_set.loc[:, features_no_time_clog + target]  = test_set[features_no_time_clog + target].interpolate(method="spline", order=3)
        
        
        train_set.drop(columns=["is_clog"], inplace=True)
        test_set.drop(columns=["is_clog"], inplace=True)
        
        train_set = self.prepare_dataframe_for_lstm(df=train_set, 
                                   n_steps=self.config.look_back)
        test_set = self.prepare_dataframe_for_lstm(df=test_set, 
                                   n_steps=self.config.look_back)
        
        train_set.loc[:, "is_rain"] = (train_set[target[0]] > 0).astype(int)
        test_set.loc[:, "is_rain"] = (test_set[target[0]] > 0).astype(int)

        
        train_set.to_csv(os.path.join(self.config.root_dir, "train.csv"))
        test_set.to_csv(os.path.join(self.config.root_dir, "test.csv"))

        logger.info("Splited data into training and test sets")
        logger.info(f"Training set {train_set.shape}")
        logger.info(f"Testin set {test_set.shape}")

