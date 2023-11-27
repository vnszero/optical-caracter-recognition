from abc import abstractmethod
from results import Result
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from typing import List,Union

class MachineLearningMethod:

    @abstractmethod
    def eval(self,df_practice:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_category:str) -> Result:
        raise NotImplementedError

class ScikitLearnMachineLearning(MachineLearningMethod):
    # ml_method is a ClassifierMixin or RegressorMixin
    # both are superclasses
    def __init__(self,ml_method:Union[ClassifierMixin,RegressorMixin]):
        self.ml_method = ml_method

    def eval(self, df_practice:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_category:str, seed:int=1) -> Result:
        # from df_practice, split features from category
        # x_practice holds features
        # y_practice holds category value
        x_practice = df_practice.drop(col_category,axis=1)
        y_practice = df_practice[col_category]

        # execute the fit method of ml_method e create the model
        model = self.ml_method.fit(x_practice,y_practice)
        # split to predict data in x and y groups
        x_to_predict = df_data_to_predict.drop(col_category,axis=1)
        y_to_predict = df_data_to_predict[col_category]

        # return results
        y_predictions = model.predict(x_to_predict)
        return Result(y_to_predict,y_predictions)
