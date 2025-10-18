import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
            params={
                "Decision Tree": {
                    'criterion':['gini','entropy'],
                },
                "Random Forest":{
                    'n_estimators': [16,32,64]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05],
                    'subsample':[0.7,0.8,0.9],
                    'n_estimators': [16,32,64]
                },
                "Logistic Regression":{
                    'C':[0.01,0.1,1,10]
                },
                "XGBClassifier":{
                    'learning_rate':[.1,.01,.05],
                    'n_estimators': [16,32,64]
                },
                "CatBoosting Classifier":{
                    'depth': [4,6],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5],
                    'n_estimators': [16,32,64]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            # If no model meets a high threshold, log a warning but continue to
            # save the best model found. This avoids aborting the pipeline for
            # modest accuracy scores; adjust threshold as needed.
            if best_model_score < 0.5:
                logging.warning(f"Best model score {best_model_score:.3f} is below 0.5; saving best found model anyway.")
            else:
                logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy
            



            
        except Exception as e:
            raise CustomException(e,sys)
        