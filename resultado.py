from sklearn.exceptions import UndefinedMetricWarning
import optuna
import numpy as np
import pandas as pd
import warnings
from typing import List

class Resultado():
    def __init__(self, y:List[float], predict_y:List[float]):
        """
        y: Vetor numpy (np.array) em que, para cada instancia i, y[i] é a classe alvo da mesma
        predict_y: Vetor numpy (np.array) que representa a predição y[i] para a instancia i

        Tanto y quando predict_y devem assumir valores numéricos
        """
        self.y = y
        self.predict_y = predict_y
        self._mat_confusao = None
        self._precisao = None
        self._revocacao = None

    @property
    def mat_confusao(self) -> np.ndarray:
        """
        Retorna a matriz de confusão.
        """
        #caso a matriz de confusao já esteja calculada, retorna-la
        if self._mat_confusao  is not None:
            return self._mat_confusao

        ## Obtem todos os valores de classes
        set_classes = set(self.y)|set(self.predict_y)
        #instancia a matriz de confusao como uma matriz de zeros
        #A matriz de confusão terá o tamanho como o máximo entre os valores de self.y e self.predict_y
        self._mat_confusao = {}
        for classe_real in set_classes:
            self._mat_confusao[classe_real] = {}
            for classe_predita in set_classes:
                self._mat_confusao[classe_real][classe_predita] = 0

        #incrementa os valores da matriz baseada nas listas self.y e self.predict_y
        for i,classe_real in enumerate(self.y):
            self._mat_confusao[classe_real][self.predict_y[i]] += 1

        #print("Predict y: "+str(self.predict_y))
        #print("y: "+str(self.y))
        #print("Matriz de confusao final :"+str(self._mat_confusao))
        return self._mat_confusao

    @property
    def precisao(self):
        """
        Precisão por classe
        """
        if self._precisao is not None:
            return self._precisao

        #inicialize com um vetor de zero usando np.zeros
        self._precisao = {}

        #para cada classe, armazene em self._precisao[classe] o valor relativo à precisão
        #dessa classe
        for classe in self.mat_confusao.keys():
            #obtnha todos os elementos que foram previstos com essa classe
            num_previstos_classe = 0
            for classe_real in self.mat_confusao.keys():
                num_previstos_classe += self.mat_confusao[classe_real][classe]

            #precisao: numero de elementos previstos corretamente/total de previstos com essa classe
            #calcule a precisão para a classe
            if num_previstos_classe!=0:
                self._precisao[classe] =  self.mat_confusao[classe][classe]/num_previstos_classe
            else:
                self._precisao[classe] = 0
                warnings.warn("Não há elementos previstos para a classe "+str(classe)+" precisão foi definida como zero.", UndefinedMetricWarning)
        return self._precisao

    @property
    def revocacao(self):
        if self._revocacao is not None:
            return self._revocacao

        self._revocacao = {}
        for classe in self.mat_confusao.keys():
            #por meio da matriz, obtem todos os elementos que são dessa classe
            num_classe = 0
            num_elementos_classe = 0
            for classe_prevista in self.mat_confusao.keys():
                num_elementos_classe += self.mat_confusao[classe][classe_prevista]

            #revocacao: numero de elementos previstos corretamente/total de elementos dessa classe
            if num_elementos_classe!=0:
                self._revocacao[classe] =  self.mat_confusao[classe][classe]/num_elementos_classe
            else:
                self._revocacao[classe] = 0
                warnings.warn("Não há elementos da classe "+str(classe)+" revocação foi definida como zero.", UndefinedMetricWarning)
        return self._revocacao

    @property
    def f1_por_classe(self):
        """
        retorna um vetor em que, para cada classe, retorna o seu f1
        """
        f1 = {}
        for classe in self.mat_confusao.keys():
            if(self.precisao[classe]+self.revocacao[classe] == 0):
                f1[classe] = 0
            else:
                f1[classe] = 2*(self.precisao[classe]*self.revocacao[classe])/(self.precisao[classe]+self.revocacao[classe])
        return f1

    @property
    def macro_f1(self):
        #Atividade 1: substitua o none...lembre-se que já foi calculado o
        #f1 por classe no atributo calculado correspondente.
        #Lembre-se de como usar atributos calculados.
        return np.average(list(self.f1_por_classe.values()))

    @property
    def acuracia(self):
        #quantidade de elementos previstos corretamente
        num_previstos_corretamente = 0
        for classe in range(len(self.mat_confusao)):
            #Atividade 1: complete o código abaixo, substituindo o None
            num_previstos_corretamente  += self.mat_confusao[classe][classe]

        return num_previstos_corretamente/len(self.y)

class Fold():
    def __init__(self,df_treino :pd.DataFrame,  df_data_to_predict:pd.DataFrame,
                col_classe:str,num_folds_validacao:int=0,num_repeticoes_validacao:int=0):
        self.df_treino = df_treino
        self.df_data_to_predict = df_data_to_predict
        self.col_classe = col_classe

        #Atividade 3(b): Inicialize o arr_folds_validacao apropriadamente
        if num_folds_validacao>0:
            self.arr_folds_validacao = self.gerar_k_folds(df_treino,num_folds_validacao,col_classe,num_repeticoes_validacao)
        else:
            self.arr_folds_validacao = []

    @staticmethod
    def gerar_k_folds(df_dados,val_k:int,col_classe:str,num_repeticoes:int=1,seed:int=1,
                    num_folds_validacao:int=0,num_repeticoes_validacao:int=1) -> List["Fold"]:
        """
        Implementar esta função de acordo com os comentários no código
        Retorna um vetor arr_folds com todos os k folds criados a partir do DataFrame df_dados

        df_dados: DataFrame com os dados a serem usados
        val_k: parametro k da validação cruzada de  k-folds
        col_classe: coluna que representa a classe
        seed: seed para a amostra aleatória
        """
        #1. especifique o número de instancias da partição teste de cada fold usando
        #...o parametro val_k
        num_instances_per_partition = len(df_dados.index)//val_k
        #folds de saida
        arr_folds = []


        for num_repeticao in range(num_repeticoes):
            #2. Embaralhe os dados: para isso, use o método sample para fazer uma amostra aleatória usando 100% dos dados. Use a seed passada como parametro
            #lembre-se que, para cada repetição, deve-se haver uma seed diferente
            #para isso, use seed+num_repeticao
            df_dados_rand = df_dados.sample(frac=1,random_state=seed+num_repeticao)

            #Impressão dos ids dos dados (exiba o print para testes)
            #print("Dados: "+str(df_dados_rand.index.values))

            #para cada fold num_fold:
            for num_fold in range(val_k):
                #2. especifique o inicio e fim do fold de teste. Caso seja o ultimo, o fim será o tamanho do vetor.
                #Use num_instances_per_partition e num_fold para deliminar o inicio e fim do teste
                ini_fold_to_predict = num_instances_per_partition*num_fold
                if num_fold < val_k-1:
                    fim_fold_to_predict = num_instances_per_partition+ini_fold_to_predict
                else:
                    fim_fold_to_predict = len(df_dados)

                #print(f"Inicio: {ini_fold_to_predict} -  Fim: {fim_fold_to_predict}")
                #3. por meio do df_dados_rand, obtenha os dados de avaliação (teste ou validação)
                df_to_predict = df_dados_rand[ini_fold_to_predict:fim_fold_to_predict]
                #print(df_to_predict)

                #4. Crie o treino por meio dos dados originais (df_dados_rand),
                #removendo os dados que serão avaliados  (df_to_predict)
                df_treino = df_dados_rand.drop(df_to_predict.index)
                #print(df_treino)

                #5. Crie o fold (objeto da classe Fold) para adicioná-lo no vetor
                fold = Fold(df_treino,df_to_predict,col_classe,num_folds_validacao,num_repeticoes_validacao)
                arr_folds.append(fold)


        #imprime o número instancias por fold (descomente para testes)
        """
        for num_repeticao in range(num_repeticoes):
            for num_fold in range(val_k):
                i = val_k*num_repeticao+num_fold
                df_treino  = arr_folds[i].df_treino
                df_to_predict  = arr_folds[i].df_data_to_predict
                qtd_treino = len(df_treino.index)
                qtd_to_predict = len(df_to_predict.index)
                print(f"Repeticao #{num_repeticao}  Fold #{num_fold} instancias no treino: {qtd_treino} teste: {qtd_to_predict}")
                print(f"\tÍndices das instancias do treino: {df_treino.index.values}")
                print(f"\tÍndices das instancias a avaliar (teste ou validação): {df_to_predict.index.values}")
                print(" ")
        """
        return arr_folds

    def __str__(self):
        return f"Treino: \n{self.df_treino}\n Dados a serem avaliados (teste ou validação): {self.df_data_to_predict}"
    def __repr__(self):
        return str(self)
