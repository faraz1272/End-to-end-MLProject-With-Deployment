## Here we will define the model training functions
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import mlflow

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
    min_r2: float  # NEW

class ModelTrainer:
    def __init__(self, config: dict | None = None):
        artifacts_dir = "artifacts"
        model_name = "model.pkl"
        min_r2 = 0.6  # your current threshold

        if config:
            data_cfg = config.get("data", {})
            artifacts_dir = data_cfg.get("artifacts_dir", artifacts_dir)
            out_cfg = config.get("output", {})
            model_name = out_cfg.get("model_filename", model_name)
            min_r2 = config.get("training", {}).get("min_r2", min_r2)

        self.model_trainer_config = ModelTrainerConfig(
            trained_model_file_path=os.path.join(artifacts_dir, model_name),
            min_r2=min_r2
        )

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=0),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "KNN Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance']
                }
                
            }

                    # --- MLflow: start a run (uses local ./mlruns by default) ---
            mlflow.set_experiment("student-score")   # creates/uses this experiment locally
            with mlflow.start_run(run_name="multi-model-grid"):
                # Log which models we considered
                mlflow.log_param("candidate_models", list(models.keys()))

                # Train/evaluate
                model_report: dict = evaluate_models(
                    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    models=models, param=params
                )

                # Log each model's R^2 as a metric
                for name, score in model_report.items():
                    mlflow.log_metric(f"r2_{name.replace(' ', '_').lower()}", float(score))

                best_model_score = max(sorted(model_report.values()))
                best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]
                best_model = models[best_model_name]

                if best_model_score < self.model_trainer_config.min_r2:
                    raise CustomException("No best model found")
                logging.info("Best model found on both training and testing dataset")

                # Save best model as before
                save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )

                # --- MLflow: log summary + artifacts ---
                mlflow.set_tag("best_model", best_model_name)
                mlflow.log_metric("r2_best", float(best_model_score))

                # Log the serialized model and (if present) the preprocessor
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)
                maybe_preproc = os.path.join(
                    os.path.dirname(self.model_trainer_config.trained_model_file_path),
                    "preprocessor.pkl"
                )
                if os.path.exists(maybe_preproc):
                    mlflow.log_artifact(maybe_preproc)

                # Compute final R^2 in the same way you already do
                predicted = best_model.predict(X_test)
                r2_value = r2_score(y_test, predicted)

                # Optionally log the final R^2 again (sanity)
                mlflow.log_metric("r2_on_holdout", float(r2_value))

                return best_model_name, r2_value

            # model_report:dict= evaluate_models(X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
            #                                    models=models, param=params)
            
            # best_model_score = max(sorted(model_report.values()))

            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]

            # best_model = models[best_model_name]

            # if best_model_score<self.model_trainer_config.min_r2:
            #     raise CustomException("No best model found")
            # logging.info(f"Best model found on both training and testing dataset")

            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )

            # predicted = best_model.predict(X_test)
            # r2_value = r2_score(y_test, predicted)

            # return best_model_name, r2_value

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    import argparse
    from src.utils import read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    args = parser.parse_args()

    config = read_yaml(args.config)

    trainer = ModelTrainer(config)
    trainer.run()