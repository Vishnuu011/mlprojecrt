from src.mlproject.logger import logging
from src.mlproject.components.data_ingestion import DataIngestion

logging.info("HI this first log")

if __name__ == '__main__':

   data_inges= DataIngestion()
   data_inges.initiate_data_ingestion()