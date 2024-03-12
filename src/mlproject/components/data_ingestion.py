import sys
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from dataclasses import dataclass
#from src.mlproject.utils import load_dataframe

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:

    def __init__(self):

        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("data ingestion started")

        try:
            data=pd.read_csv("/workspaces/mlprojecrt/experiments/data/gemstone.csv")
            logging.info("loaded gemstone dataset")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("raw dataset saved in atifacts")

            train_data, test_data=train_test_split(data, test_size=0.25)
            logging.info("dataset splitted in train and test")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("train and test saved in artifacts")

            logging.info("data ingestion completed..........")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )  

        except Exception as e:
            logging.info("exception durning occured data ingestion stage")
            raise CustomException(e,sys)     



if __name__=="__main__":
    obj=DataIngestion()

    obj.initiate_data_ingestion()
    

     