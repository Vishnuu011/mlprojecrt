from src.mlproject.components.data_ingestion import DataIngestion

from dataclasses import dataclass


@dataclass
class Training_Pipline:

    def start(self):
        data_ingestion= DataIngestion()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()


if __name__ == '__main__':
    training_pipline= Training_Pipline()
    training_pipline.start()