from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation

logging.info("HI this first log")

if __name__ == '__main__':

   data_inges= DataIngestion()
   train_data_path, test_data_path=data_inges.initiate_data_ingestion()

   data_transfom=DataTransformation()
   data_transfom.initialize_data_transformation(train_data_path, test_data_path)