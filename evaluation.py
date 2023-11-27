import numpy as np
import warnings
from abc import abstractmethod
import optuna
import numpy as np
from typing import List,Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from results import Fold, Result
from method import MachineLearningMethod, ScikitLearnMachineLearning

class Experiment():
    def __init__(self,folds:List[Fold], ml_method:MachineLearningMethod,
                    goal_category_optimization=None,
                    num_trials:int=100, sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=10)):
        """
        folds: folds defined for experiments
        ml_method: machile learning method to be used
        goal_category_optimization: objective class as optimization criterion for features
        """
        self.folds = folds
        self._results = None
        self.ml_method = ml_method
        self.goal_category_optimization = goal_category_optimization
        self.num_trials = num_trials
        self.sampler = sampler
        self.studies_per_fold = []

    @property
    def results(self) -> List[Result]:
        if self._results:
            return self._results
        return self.calculate_results()

    def calculate_results(self)  -> List[Result]:
        """
        Retorns, for each fold, its result
        """
        self._results = []
        self.arr_validation_per_fold = [] # experiments de validation per fold
        # seed to keep experiment remakeble
        np.random.seed(1)
        for i,fold in enumerate(self.folds):
            if(self.goal_category_optimization is not None):
                study = optuna.create_study(sampler=self.sampler, direction='maximize')
                optimization_goal = self.goal_category_optimization(fold)
                study.optimize(optimization_goal, n_trials=self.num_trials)

                best_method = optimization_goal.arr_evaluated_methods[study.best_trial.number]
                self.studies_per_fold.append(study)
            else:
                best_method = self.ml_method

            result = best_method.eval(fold.df_practice,fold.df_data_to_predict,fold.col_category)
            self._results.append(result)

        return self._results

    @property
    def macro_f1_avg(self) -> float:
        """
        Calculate average f1 of results.
        """
        return np.mean([result.macro_f1 for result in self.results])

class GoalOptimization:
    def __init__(self,  fold: Fold):
        self.fold = fold
        self.arr_evaluated_methods = []

    @abstractmethod
    def get_method(self,trial: optuna.Trial) ->MachineLearningMethod:
        raise NotImplementedError

    @abstractmethod
    def optimization_result(self,result:Result) -> float:
        raise NotImplementedError

    def __call__(self, trial: optuna.Trial) -> float:
        # for each fold, execute method and calculate result
        sum = 0
        method = self.get_method(trial)
        self.arr_evaluated_methods.append(method)
        for fold_validation in self.fold.arr_folds_validation:
            result = method.eval(fold_validation.df_practice,fold_validation.df_data_to_predict,self.fold.col_category)
            sum += self.optimization_result(result)

        return sum/len(self.fold.arr_folds_validation)

class GoalOptimizationDecisionTree(GoalOptimization):
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def get_method(self,trial: optuna.Trial) -> MachineLearningMethod:

        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        clf_dtree = DecisionTreeClassifier(min_samples_split=min_samples,random_state=2)

        return ScikitLearnMachineLearning(clf_dtree)

    def optimization_result(self,result):
        return result.macro_f1

class GoalOptimizationRandomForest(GoalOptimization):
    def __init__(self, fold:Fold, num_max_trees:int=5):
        super().__init__(fold)
        self.num_max_trees = num_max_trees

    def get_method(self,trial: optuna.Trial)->MachineLearningMethod:        
        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        max_features = trial.suggest_uniform('max_features', 0, 0.5)
        num_trees = trial.suggest_int('num_trees', 1, self.num_max_trees)
        clf_rf = RandomForestClassifier(min_samples_split=min_samples,max_features=max_features,n_estimators=num_trees,random_state=2)
        
        return ScikitLearnMachineLearning(clf_rf)

    def optimization_result(self, result:Result) ->float:
        return result.macro_f1
