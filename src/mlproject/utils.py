import os
import sys
import pandas as pd
#from pathlib import Path
import numpy as np
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pickle

#def load_dataframe(path:str, file_name:str):
    #return pd.read_csv(Path(os.path.join(path, file_name)))

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
