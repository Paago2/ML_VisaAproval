import sys
from typing import Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,roc_auc_score
from neuro_mf import ModelFactory

from visa_approval.exception import VisaApprovalException
from visa_approval.logger import logging
from visa_approval.entity.artifact_entity import ModelTrainerArtifact,ClassificationMetricsArtifact, DataTransformationArtifact
from visa_approval.entity.estimator import VisaApprovalModel
from visa_approval.utils.main_utils import save_object ,read_yaml_file, load_numpy_array_data, load_object
from visa_approval.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self,train:np.array,test:np.array)-> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This method uses neuro_mf to get the best model object and report of the best model
        Output      :   Returns the metric artifact and the best model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("using neuro_mf to get the best model object and report of the best model")
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)

            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            best_model_detail = model_factory.get_best_model(X = x_train, y = y_train, base_accuracy= self.model_trainer_config.expected_score
            )
            model_object = best_model_detail.best_model

            y_pred = model_object.predict(x_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            metric_artifact = ClassificationMetricsArtifact(
                                                    accuracy_score=accuracy,
                                                    precision_score=precision,
                                                    recall_score=recall,
                                                    f1_score=f1,
                                                    roc_auc=roc_auc
                                                    )
            return best_model_detail, metric_artifact
        
        except Exception as e:
            raise VisaApprovalException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Method Name :   initiate_model_trainer
        Description :   This method is responsible for initiating the model trainer steps
        Output      :   Returns the model training artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_object_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            best_model_detail, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            preprocessing_object = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_score:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")
            visa_approval_model = VisaApprovalModel(
                preprocessing_object=preprocessing_object,
                trained_best_model=best_model_detail.best_model
            )
            logging.info("Created visa approval model object with preprocessor and model")
            logging.info("Created best model file path")
            save_object(file_path=self.model_trainer_config.model_file_path, object=visa_approval_model)

            logging.info("Saved the best model object")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.model_file_path,
                metric_artifact=metric_artifact
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise VisaApprovalException(e, sys) from e