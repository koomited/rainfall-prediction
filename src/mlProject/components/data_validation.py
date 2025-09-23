import pandas as pd
import warnings
import os
from mlProject.entity.config_entity import DataValidationConfig
from mlProject import logger

class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)
            dict_cols_types = data.dtypes.apply(lambda x: x.name).to_dict()

    
            features = self.config.all_schema.COLUMNS
            target = self.config.all_schema.TARGET_COLUMN
            all_schema = {**features, **target}

            
            for col in all_schema:
                try:
                    if dict_cols_types[col]==all_schema[col]:
                        validation_status = True
                    else:
                        validation_status = False
                        break
                except:
                    validation_status = False
                    
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            return validation_status
        
        except Exception as e:
            raise e