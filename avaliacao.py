import numpy as np
import warnings
from abc import abstractmethod
import optuna
import numpy as np
from typing import List,Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from resultado import Fold,Resultado
from metodo import MetodoAprendizadoDeMaquina,ScikitLearnAprendizadoDeMaquina

class Experimento():
    def __init__(self,folds:List[Fold], ml_method:MetodoAprendizadoDeMaquina,
                    ClasseObjetivoOtimizacao=None,
                    num_trials:int=100, sampler=optuna.samplers.TPESampler(seed=1, n_startup_trials=10)):
        """
        folds: folds a serem usados no experimentos
        ml_method: Método de aprendizado de máquina a ser usado
        ClasseObjetivoOtimizacao: CLASSE a ser usada para otimização dos parametros
        """
        self.folds = folds
        self._resultados = None
        self.ml_method = ml_method
        self.ClasseObjetivoOtimizacao = ClasseObjetivoOtimizacao
        self.num_trials = num_trials
        self.sampler = sampler
        self.studies_per_fold = []

    @property
    def resultados(self) -> List[Resultado]:
        if self._resultados:
            return self._resultados
        return self.calcula_resultados()

    def calcula_resultados(self)  -> List[Resultado]:
        """
        Atividade 5: Complete o código abaixo substituindo os "None", quando necessário
        Retorna, para cada fold, o seu respectivo resultado
        """
        self._resultados = []
        self.arr_validacao_por_fold = [] #experimentos de validacao por fold
        #seed para mater a reprodutibilidade dos experimentos
        np.random.seed(1)
        ## Para cada fold
        for i,fold in enumerate(self.folds):
            ##1. Caso haja um metodo de otimizacao, obtenha o melhor metodo com ele
            #substitua os none quando necessario
            if(self.ClasseObjetivoOtimizacao is not None):
                study = optuna.create_study(sampler=self.sampler, direction='maximize')
                objetivo_otimizacao = self.ClasseObjetivoOtimizacao(fold)
                study.optimize(objetivo_otimizacao, n_trials=self.num_trials)

                #1.(a) obtem o melhor metodo da otimização
                # . use o vetor arr_evaluated_methods e o número do best_trial (study.best_trial.number)
                best_method = objetivo_otimizacao.arr_evaluated_methods[study.best_trial.number]
                self.studies_per_fold.append(study)
            else:
                #caso contrario, o metodo, atributo da classe Experimento (sem modificações) é usado
                best_method = self.ml_method

            #2. Efetua a predição nos valores de teste (fold.df_data_to_predict)
            # . logo após, adiciona em resultados o resultado predito (objeto da classe Resultado) usando o melhor metodo
            resultado = best_method.eval(fold.df_treino,fold.df_data_to_predict,fold.col_classe)
            self._resultados.append(resultado)

        return self._resultados

    @property
    def macro_f1_avg(self) -> float:
        """
        Calcula a média do f1 dos resultados.
        """
        return np.mean([resultado.macro_f1 for resultado in self.resultados])

class OtimizacaoObjetivo:
    def __init__(self,  fold: Fold):
        self.fold = fold
        self.arr_evaluated_methods = []

    @abstractmethod
    def obtem_metodo(self,trial: optuna.Trial) ->MetodoAprendizadoDeMaquina:
        raise NotImplementedError

    @abstractmethod
    def resultado_metrica_otimizacao(self,resultado:Resultado) -> float:
        raise NotImplementedError

    def __call__(self, trial: optuna.Trial) -> float:
        #para cada fold, executa o método e calcula o resultado
        sum = 0
        metodo = self.obtem_metodo(trial)
        self.arr_evaluated_methods.append(metodo)
        for fold_validacao in self.fold.arr_folds_validacao:
            resultado = metodo.eval(fold_validacao.df_treino,fold_validacao.df_data_to_predict,self.fold.col_classe)
            sum += self.resultado_metrica_otimizacao(resultado)

        return sum/len(self.fold.arr_folds_validacao)

class OtimizacaoObjetivoArvoreDecisao(OtimizacaoObjetivo):
    def __init__(self, fold:Fold):
        super().__init__(fold)

    def obtem_metodo(self,trial: optuna.Trial) -> MetodoAprendizadoDeMaquina:

        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        clf_dtree = DecisionTreeClassifier(min_samples_split=min_samples,random_state=2)

        return ScikitLearnAprendizadoDeMaquina(clf_dtree)

    def resultado_metrica_otimizacao(self,resultado):
        return resultado.macro_f1

class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, num_arvores_max:int=5):
        super().__init__(fold)
        self.num_arvores_max = num_arvores_max

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Atividade 4: complete este método
        # Para passar nos testes, os parametros devem ter o seguintes nomes: "min_samples_split",
        #. "max_features" e "num_arvores". Não mude a ordem de atribuição
        #. abaixo
        min_samples = trial.suggest_uniform('min_samples_split', 0, 0.5)
        max_features = trial.suggest_uniform('max_features', 0, 0.5)
        num_arvores = trial.suggest_int('num_arvores', 1, self.num_arvores_max)
        #coloque, ao instanciar o RandomForestClassifier como random_state=2
        clf_rf = RandomForestClassifier(min_samples_split=min_samples,max_features=max_features,n_estimators=num_arvores,random_state=2)

        return ScikitLearnAprendizadoDeMaquina(clf_rf)

    def resultado_metrica_otimizacao(self, resultado:Resultado) ->float:
        #Atividade 4: calcule o resultado por meio do macro_f1
        return resultado.macro_f1
