import sys
from pandas import DataFrame

from visa_approval.exception import VisaApprovalException
from visa_approval.logger import logging
from sklearn.pipeline import Pipeline


class TargetValueMapping:
    def __init__(self):
        self.Certified: int = 1
        self.Denied: int = 0
    def _asdict(self):
        return self.__dict__
    def reverse_mapping(self):
        mapping_response = self._asdict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))
    
class VisaApprovalModel:
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocessor 
        :param trained_model_object: Input Object of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

 

class VisaApprovalModel:
    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transforms raw input using preprocessing_object
        to ensure that the inputs are in the same format as the training data.
        Finally, it performs prediction on transformed features.
        """
        logging.info("Entered the predict method of VisaApprovalModel class")
        
        try:
            logging.info("Using the trained model to make predictions")
            
            transformed_data = self.preprocessing_object.transform(dataframe)
            logging.info("Used the trained model to make predictions")
            
            return self.trained_model_object.predict(transformed_data)
        
        except Exception as e:
            raise VisaApprovalException(e, sys) from e
    
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
