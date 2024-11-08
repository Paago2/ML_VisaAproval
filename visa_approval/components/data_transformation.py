import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder,PowerTransformer
from sklearn.compose import ColumnTransformer

from visa_approval.exception import VisaApprovalException
from visa_approval.logger import logging
from visa_approval.utils.main_utils import read_yaml_file, drop_columns, save_object, save_numpy_array_data
from visa_approval.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from visa_approval.entity.config_entity import DataTransformationConfig
from visa_approval.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from visa_approval.entity.estimator import TargetValueMapping



class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                    data_validation_artifact: DataValidationArtifact,
                    data_transformation_artifact: DataTransformationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_artifact: Output reference of data validation artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise VisaApprovalException(e, sys)
            
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VisaApprovalException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method returns the data transformer object
        Output      :   Returns the transformer object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered the get_data_transformer_object method of DataTransformation class")
            
            numerical_transformer = StandardScaler()
            one_hot_encoder_transformer = OneHotEncoder(handle_unknown="ignore")
            ordinal_encoder = OrdinalEncoder()
            
            logging.info("Initialize Standerescaler, OneHotEncoder and OrdinalEncoder")
            one_hot_encoder_transformer = self._schema_config["oh_columns"]
            ordinal_encoder = self._schema_config["ordinal_columns"]
            transform_columns = self._schema_config["transform_columns"]
            numerical_features = self._schema_config["num_features"]
            
            logging.info("Initialize PowerTransformer")

            transform_pipeline = Pipeline(steps=[
                ("transformer", PowerTransformer(method="yeo-johnson"))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ("Standardization", numerical_transformer, numerical_features),
                    ("OneHotEncoder", one_hot_encoder_transformer, transform_columns),
                    ("Ordinal_Encoder", ordinal_encoder, transform_columns),
                    ("Transformer", transform_pipeline, transform_columns)
                ]
                )
            logging.info("Created Object from ColumnTransformer")

            logging.info("Exited the get_data_transformer_object method of DataTransformation class")
            return preprocessor
        
        except Exception as e:
            raise VisaApprovalException(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name :   initiate_data_transformation
        Description :   This method is used to initiate the data transformation component for the pipeline
        Output      :   data transformer steps are performed and preprocessor object is created
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Staring data transformation")
                
                preprocessor = self.get_data_transformer_object()
                logging.info("Got the preprocessor object")

                train_df = DataTransformation.read_data(self.data_ingestion_artifact.train_file_path)
                test_df = DataTransformation.read_data(self.data_ingestion_artifact.test_file_path)


                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                logging.info("Got trained features and test features")

                input_feature_train_df["company_age"] = CURRENT_YEAR - input_feature_train_df["yr_of_estab"]

                logging.info("Added company age column to training dataset")

                drop_cols = self._schema_config["drop_columns"]
                logging.info("Dropping columns of drop_cols from training dataset")

                input_feature_train_df = drop_columns(data=input_feature_train_df, columns=drop_cols)

                target_feature_train_df = target_feature_train_df.replace       (TargetValueMapping().asdict()
                )

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]

                input_feature_test_df["company_age"] = CURRENT_YEAR - input_feature_test_df["yr_of_estab"]

                logging.info("Added company age column to testing dataset")

                input_feature_test_df = drop_columns(data=input_feature_test_df, columns=drop_cols)

                logging.info("Dropped columns of drop_cols from testing dataset")

                target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().asdict())

                logging.info("Got training features and test features of Testing dataset")

                logging.info("Applying the preprocessor object on training dataframes and test dataframes")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)

                logging.info("Used the preprocessor object to transform the test feaatures ")    

                input_feature_test_arr = preprocessor.transform(input_feature_test_df)  

                logging.info("Used preprocessor object to transform the test features")

                logging.info("applying SMOTEENN on training features")

                smote_enn = SMOTEENN(sampling_strategy= "minority") 

                input_feature_train_final, target_feature_train_final = smote_enn.fit_resample(input_feature_train_arr, target_feature_train_df)  

                logging.info("Applied SMOTEENN on training dataset")  

                logging.info("Applying SMOTEENN on test features")

                input_feature_test_final, target_feature_test_final = smote_enn.fit_resample(input_feature_test_arr, target_feature_test_df)

                logging.info("Applied SMOTEENN on test dataset")

                logging.info("created train array and test array")

                train_array = np.c_[
                    input_feature_train_final, np.array(target_feature_train_final)
                    ] 
                
                test_array = np.c_[
                    input_feature_test_final, np.array(target_feature_test_final)
                    ]
                
                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_array)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_array)

                logging.info("Saved the preprocessor object")

                logging.info("Exited the initiate_data_transformation method of DataTransformation class")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path= self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path= self.data_transformation_config.transformed_test_file_path
                )

                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
            
        except Exception as e:
            raise VisaApprovalException(e, sys) from e
                                       
                                


        