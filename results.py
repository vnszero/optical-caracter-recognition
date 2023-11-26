from sklearn.exceptions import UndefinedMetricWarning
import optuna
import numpy as np
import pandas as pd
import warnings
from typing import List

class Result():
    def __init__(self, y:List[float], predict_y:List[float]):
        """
        y: Vetor numpy (np.array) target class y[i] for each i
        predict_y: Vetor numpy (np.array) predicted y[i] for each i

        y and predict_y must hold numeric values
        """
        self.y = y
        self.predict_y = predict_y
        self._confusion_matrix = None
        self._precision = None
        self._recall = None

    @property
    def confusion_matrix(self) -> np.ndarray:
        """
        Return confusion matrix
        """
        # just return if it already exists
        if self._confusion_matrix  is not None:
            return self._confusion_matrix

        # get all categorys values
        set_class = set(self.y)|set(self.predict_y)
        # let's start confusion matrix as a zero matrix
        # confusion matrix will have a maximum size based on values of self.y and self.predict_y
        self._confusion_matrix = {}
        for real_category in set_class:
            self._confusion_matrix[real_category] = {}
            for class_predita in set_class:
                self._confusion_matrix[real_category][class_predita] = 0

        # the values of the matrix should be incresed based on self.y and self.predict_y lists
        for i,real_category in enumerate(self.y):
            self._confusion_matrix[real_category][self.predict_y[i]] += 1

        return self._confusion_matrix

    @property
    def precision(self):
        """
        precision per class
        """
        if self._precision is not None:
            return self._precision

        self._precision = {}

        # for each class, it will store the relative value of precision en self._precision[class]
        for category in self.confusion_matrix.keys():
            # get all elements which have been predicted for this class
            num_predicted_category = 0
            for real_category in self.confusion_matrix.keys():
                num_predicted_category += self.confusion_matrix[real_category][category]

            # precision: number of elements predicted correctly / all predicted with this category
            # calculate precision for this category
            if num_predicted_category!=0:
                self._precision[category] =  self.confusion_matrix[category][category]/num_predicted_category
            else:
                self._precision[category] = 0
                warnings.warn("There is no predicted elements for this category "+str(category)+" precisio set as zero.", UndefinedMetricWarning)
        return self._precision

    @property
    def recall(self):
        if self._recall is not None:
            return self._recall

        self._recall = {}
        for category in self.confusion_matrix.keys():
            # pass through matrix, get all elements for this category
            num_category = 0
            num_elements_category = 0
            for predicted_category in self.confusion_matrix.keys():
                num_elements_category += self.confusion_matrix[category][predicted_category]

            # recall: number of correctly predicted elements / total o elements from this category
            if num_elements_category!=0:
                self._recall[category] =  self.confusion_matrix[category][category]/num_elements_category
            else:
                self._recall[category] = 0
                warnings.warn("There is no elemenst for this category "+str(category)+" recall set as zero.", UndefinedMetricWarning)
        return self._recall

    @property
    def f1_per_category(self):
        """
        returns a vector for each category's f1
        """
        f1 = {}
        for category in self.confusion_matrix.keys():
            if(self.precision[category]+self.recall[category] == 0):
                f1[category] = 0
            else:
                f1[category] = 2*(self.precision[category]*self.recall[category])/(self.precision[category]+self.recall[category])
        return f1

    @property
    def macro_f1(self):
        # f1 per category over calculated feature.
        return np.average(list(self.f1_per_category.values()))

    @property
    def accuracy(self):
        # number of correctly predicted elements
        num_previstos_corretamente = 0
        for category in range(len(self.confusion_matrix)):
            num_previstos_corretamente  += self.confusion_matrix[category][category]

        return num_previstos_corretamente/len(self.y)

class Fold():
    def __init__(self,df_practice :pd.DataFrame,  df_data_to_predict:pd.DataFrame,
                col_category:str,num_folds_validation:int=0,num_threshold_validation:int=0):
        self.df_practice = df_practice
        self.df_data_to_predict = df_data_to_predict
        self.col_category = col_category

        # Start arr_folds_validation properly
        if num_folds_validation>0:
            self.arr_folds_validation = self.generate_k_folds(df_practice,num_folds_validation,col_category,num_threshold_validation)
        else:
            self.arr_folds_validation = []

    @staticmethod
    def generate_k_folds(df_data,val_k:int,col_category:str,num_threshold:int=1,seed:int=1,
                    num_folds_validation:int=0,num_threshold_validation:int=1) -> List["Fold"]:
        """
        Return a vector arr_folds with all created k folds based on DataFrame df_data

        df_data: DataFrame with all data to be used
        val_k: parameter k of cross validation for k-folds
        col_category: column that represents a category
        seed: seed for random sample
        """
        
        num_instances_per_partition = len(df_data.index)//val_k
        # output folds
        arr_folds = []


        for index in range(num_threshold):
            # random sample
            df_data_rand = df_data.sample(frac=1,random_state=seed+index)

            # for each num_fold:
            for num_fold in range(val_k):
                # num_instances_per_partition and num_fold to define the begin and end of test
                ini_fold_to_predict = num_instances_per_partition*num_fold
                if num_fold < val_k-1:
                    fim_fold_to_predict = num_instances_per_partition+ini_fold_to_predict
                else:
                    fim_fold_to_predict = len(df_data)

                # set a df to predict
                df_to_predict = df_data_rand[ini_fold_to_predict:fim_fold_to_predict]

                # set practice
                df_practice = df_data_rand.drop(df_to_predict.index)

                # set fold and store it
                fold = Fold(df_practice,df_to_predict,col_category,num_folds_validation,num_threshold_validation)
                arr_folds.append(fold)

        return arr_folds

    def __str__(self):
        return f"Practice: \n{self.df_practice}\n Data to be used (test or validation): {self.df_data_to_predict}"
    def __repr__(self):
        return str(self)
