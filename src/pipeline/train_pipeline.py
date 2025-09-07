import sys
import os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException



# TrainingPipeline Class

# The TrainingPipeline class orchestrates the entire
# machine learning pipeline by running the components
# sequentially: data ingestion,data transformation,
# and model training

class TrainingPipeline:


#     start_data_ingestion() Method

# This method indicates the data ingestion process,
# which is responsible for fetching data from a source
# (eg,a database,CSV file)

# Steps
# 1.An instance of DataIngestion is created 
# 2.The initiate_data_ingestion() method of DataIngestion
# is called, which ingests the data and stores it in a
# "feature store" (a structured file or database)
# 3. The path to the feature store file (where the data
# is saved) is returned.

    def start_data_ingestion(self):
        try:
            data_ingestion= DataIngestion()
            feature_store_file_path=data_ingestion.initiate_data_ingestion()

            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
        

#         start_data_transformation() method

# The method initiates the data transformation process,
# which is responsible for preprocessing the data
# (eg,scaling encoding) and splitting it into training 
# and testing sets

# Steps
# 1.An instance of DataTransformation is created with 
# the feature store file path passed to it.

# 2.The initiate_data_transformation() method of DataTransformation
# is called,which transforms the data and splits it into
# training and test sets

# 3.The method returns
#     a.train_arr:The transformed training data
#     b.test_arr:The transformed test data
#     c.preprocessor_path: The path where the preprocessor
#     (for scaling,imputing etc) is saved

    def start_data_transformation(self,feature_store_file_path):
        try:
            data_transformation=DataTransformation(feature_store_file_path=feature_store_file_path)
            train_arr,test_arr,preprocessor_path=data_transformation.initiate_data_transformation()

            return train_arr,test_arr,preprocessor_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    
    # start_model_training() Method

    # This method initiates the model training process,
    # which is responsible for training machine learning 
    # models and evaluating their performance.

    # Steps:
    # 1.An instance of ModelTrainer is created

    # 2.The initiate_model_train() method of ModelTrainer is
    # called,passing the training and test data arrays(train_arr,test_arr)

    # 3.The model is trained and the final model score
    # (such ar r2_score or accuracy) is returned

    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_score=model_trainer.initiate_model_train(train_arr=train_arr,test_arr=test_arr)

            return model_score
        
        except Exception as e:
            raise CustomException(e,sys)
        

    #         run_pipeline() Method

    # This is the main method that runs the entire machine 
    # learning pipeline,executing data ingestion,
    # transformation,and model training in sequence

    # Steps
    # 1.Data Ingestion: Calls start_data_ingestion() to ingest
    # the data and get the feature store file path

    # 2.Data Transformation: Calls start_data_transformation()
    # to preprocess the data and split it into training and 
    # test sets

    # 3.Model Training: Calls start_model_training() to train the
    # model and get its score

    # 4.The final model score is printed to the console after training is
    # completed


    def run_pipeline(self):
        try:
            feature_store_file_path=self.start_data_ingestion()
            train_arr,test_arr,preprocessor_path=self.start_data_transformation(feature_store_file_path=feature_store_file_path)
            r2_square=self.start_model_training(train_arr=train_arr,test_arr=test_arr)

            print(f"Model training completed successfully. Model score: {r2_square}")

        except Exception as e:
            raise CustomException(e,sys)


    
    







