import os
import sys
import dill

import pandas as pd
import numpy as np

from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function saves a Python object to a specified file path using dill serialization.
    Args:
        file_path (str): The path where the object will be saved.
        obj (object): The Python object to be saved.
    Raises:
        CustomException: If there is an error during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)