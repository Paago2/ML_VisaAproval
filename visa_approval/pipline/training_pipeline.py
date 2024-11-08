import sys
from visa_approval.exception import VisaApprovalException
from visa_approval.logger import logging
from visa_approval.components.data_ingestion import DataIngestion
from visa_approval.components.data_validation import DataValidation
from visa_approval.components.data_transformation import DataTransformation
from visa_approval.components.model_trainer import ModelTrainer


from visa_approval.entity.config_entity import (
                                                    DataIngestionConfig, 
                                                    DataValidationConfig, 
                                                    DataTransformationConfig,
                                                    ModelTrainerConfig
                                                    )

from visa_approval.entity.artifact_entity import (
                                                    DataIngestionArtifact, 
                                                    DataValidationArtifact, 
                                                    DataTransformationArtifact,
                                                    ModelTrainerArtifact
                                                    )


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise VisaApprovalException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component
        """
        logging.info(
            "Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")

            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise VisaApprovalException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component
        """
        logging.info(
            "Entered the start_data_transformation method of TrainPipeline class")

        try:
            data_transformation = DataTransformation(
                        data_ingestion_artifact=self.data_ingestion_artifact,
                        data_validation_artifact=self.data_validation_artifact
                        )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise VisaApprovalException(e, sys) from e
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        This method of TrainPipeline class is responsible for starting model trainer component
        """
        logging.info(
            "Entered the start_model_trainer method of TrainPipeline class")

        try:
            model_trainer = ModelTrainer(
                        data_transformation_artifact=data_transformation_artifact, model_trainer_config=self.model_trainer_config
                        )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise VisaApprovalException(e, sys) from e 

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)

        except Exception as e:
            raise VisaApprovalException(e, sys)
