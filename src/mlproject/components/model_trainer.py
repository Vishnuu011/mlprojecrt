import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression,  Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_trainer_config=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    
    def __init__(self):

        self.trained_model_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("splitting training and testing input data.....")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Rrandom forest":RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
            }
            
                
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            print(model_report)
            logging.info(f'Model Report : {model_report}')
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.trained_model_config.trained_model_trainer_config,
                 obj=best_model
            )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)        
            
            