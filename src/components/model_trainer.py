import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

from dataclasses import dataclass


'''
This class defines the configuration for the
model training process


artifact_folder:directory where artifacts
(like trained models) are stored

trained_model_path:path where the final trained model
will be saved

expected_accuracy:The minimum expected accuracy for 
the model

model_config_file_path: Path to the configuration 
file (model.yaml) that conatains the model 
hyperparameters

'''

@dataclass
class ModelTrainerConfig:




    artifact_folder=os.path.join(artifact_folder)
    trained_model_path=os.path.join(artifact_folder,"model.pkl")
    excepted_accuracy=0.45

    # here config is the folder name 
    model_config_file_path=os.path.join('config','model.yaml')



'''
ModelTrainer Class

This is the main class responsible for 
training,evaluating,and fine-tuning models to 
find the best one for the given dataset

The main purpose of this class is to initialize the
ModelTrainer class

Attributes

model_trainer_config: Instance of ModelTrainerConfig,
which contains the configuration for the model
training process

utils: Instance of MainUtils,which provides utility
functions like saving objects

models: Dictionary of machine learning models(XGBoost,Gradient Boosting, SVC, and Random Forest)
that will be trained and evaluated

'''


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

        self.utils=MainUtils()

        # Models used for training 
        self.models={
            'XGBClassifier':XGBClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'SVC':SVC(),
            'RandomForestClassier':RandomForestClassifier()
        }


# evaluate_models() Method

# This method trains and evaluates each model in the models
# dictionary on the training data,then calculates accuracy for
# both the training and test sets

# Parameters
# X: The feature matrix
# y: The target vector
# models : Dictionary of models to evaluate

# Process
# 1.Split the data into training and test sets
# using train_test_split()

# 2.For each model in the models dictionary
#     a.Train the model on the training data
#     b.Predict the labels for both the training and test sets.
#     c.Calculate accuracy for both training and test sets using accuracy_score()
#     d.Store the test accuracy in the report dictionary

# 3. return: A dictionary(report) with model names as keys and their test accuracies as values


    def evaluate_models(self,X,y,models):
        try:
            X_train,X_test,y_train,y_test=train_test_split(
                X,y,test_size=0.2,random_state=42
            )

            report={}

            for i in range(len(list(models))):
                model=list(models.values())[i]

                # Train model
                model.fit(X_train,y_train)

                y_train_pred=model.predict(X_train)

                y_test_pred=model.predict(X_test)

                train_model_score=accuracy_score(y_train,y_train_pred)

                test_model_score=accuracy_score(y_test,y_test_pred)

                report[list(models.keys())[i]]=test_model_score

            return report
        
        except Exception as e:
            raise CustomException(e,sys)

# get_best_model() Method

# The purpose of this method is to find the best model 
# based on the accuracy score from evaluate_models()

# Parameters
# X_train,y_train,X_test,y_test:Feature and target arrays
# for training and testing

# Steps
# 1.Calls evaluate_models() to evaluate the models on
# the training data

# 2.Finds the model with the highest test accuracy from
# the model_report

# 3.Returns the best model's name,object and score

    def get_best_model(self,
                       X_train:np.array,
                       y_train:np.array,
                       X_test:np.array,
                       y_test:np.array):
        
        try:

            model_report:dict = self.evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                models=self.models
            )

            print(model_report)

            best_model_score=max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model_object=self.models[best_model_name]

            return best_model_name,best_model_object,best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)
        

    '''

finetune_best_model() Method

The purpose of this method is to fine-tune the best
model using GridSearchCV to search for the best 
hyperparameters

Parameters
a.best_model_object: The best model(selected from get_best_model())
b.best_model_name: The name of the best model
c.X_train,y_train : Training data used for fine-tuning

Steps
1. Reads the model's hyperparameter grid from a YAML
file using the MainUtils.read_yaml_file() method

2.Performs a grid search on the model to find the best
hyperparameters

3.Updates the best model with the fine-tuned parameters
and returns the fine-tuned model.


'''

        
    def finetune_best_model(self,
                            best_model_object:object,
                            best_model_name,
                            X_train,
                            y_train,) ->object:
        
        try:
            model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name] ["search_param_grid"]

            grid_search=GridSearchCV(
                best_model_object,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1
            )

            grid_search.fit(X_train,y_train)

            best_params=grid_search.best_params_

            print("best params are:",best_params)

            finetuned_model=best_model_object.set_params(**best_params)


            return finetuned_model
        
        except Exception as e:
            raise CustomException(e,sys)
        
        '''
initiate_model_trainer() method
Purpose:The main method that orchestrates the entire
model training,evaluation and fine tuning process

Parameters
train_array,test_array:Arrays containing training and test data

Steps
1.Split the input arrays into features(X) and 
target(y) for both training and testing sets

2.Calls evaluate_models() to get the performance
of each model

3.Identifies the best model using get_base_model()

4.Fine-tunes the best model using finetune_best_model()

5.Trains the fine-tuned model on the training set and
evaluates its performance on the test set

6.If the model meets the accuracy threshold,it saves the model
to disk as a .pkl file

7.return:The path to the saved model

'''


        def initiate_model_trainer(self,train_array,test_array):
            try:
                logging.info(f"Splitting training and testing input and target feature")

                X_train,y_train,X_test,y_test=(
                    train_array[:,:-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1],
                )

                logging.info(f"Extracing model config file path")

                model_report:dict=self.evaluate_models(X=X_train,y=y_train,models=self.models)

                # To get best model score from dict
                best_model_score=max(sorted(model_report.values()))

                # To get the best model name from dict 

                best_model_name=list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                ]

                best_model=self.models[best_model_name]

                best_model=self.finetune_best_model(
                    best_model_nam=best_model_name,
                    best_model_object=best_model_object,
                    X_train=X_train,
                    y_train=y_train
                )


            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
           
            print(f"best model name {best_model_name} and score: {best_model_score}")




            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
           
            logging.info(f"Best found model on both training and testing dataset")


 
       


            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_path}"
            )


            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)


            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
           
            return self.model_trainer_config.trained_model_path


           


           


        except Exception as e:
            raise CustomException(e, sys)










