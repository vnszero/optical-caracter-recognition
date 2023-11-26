import numpy as np
import pandas as pd
import hiplot as hip
from avaliacao import Experimento

def matrix_recover(all_data:pd.DataFrame, to_show:int, square_reference:int) -> list:
    line_breaker = 0
    matrix_row = 0
    matrix = []
    matrix.append([])

    # pass through all_data
    for index, row in all_data.iterrows():
        
        # the choosen one
        if index == to_show:
            
            # recover itens
            for item in row:
                
                # filter the y_class item
                if matrix_row < square_reference:
                    
                    # build matrix
                    matrix[matrix_row].append(item)
                    line_breaker += 1
                    
                    # new line
                    if (line_breaker%square_reference == 0):
                        matrix_row += 1
                        
                        # append new line except for the last one
                        if matrix_row < square_reference:
                            matrix.append([])
                            
                        line_breaker = 0
            
            # No need to run iterrows until the end
            break

    return matrix

def show_results(name:str, numbers:dict, exp:Experimento) -> None:
    print(name)
    for resultado in exp.resultados:
        print("Macro f1: " + str(resultado.macro_f1))
        print("\nAcuracia: " + str(resultado.acuracia))
        print("\nf1 por classe: ")
        for key in resultado.f1_por_classe.keys():
            print('\t{} -> {}'.format(numbers[key], resultado.f1_por_classe[key]))
        print("\nprecisao: ")
        for key in resultado.precisao.keys():
            print('\t{} -> {}'.format(numbers[key], resultado.precisao[key]))
        print("\nrevocacao: ")
        for key in resultado.revocacao.keys():
            print('\t{} -> {}'.format(numbers[key], resultado.revocacao[key]))
        print('\nMatriz de confusao: ')
        print("\t", end='')
        for names in numbers.values():
            print(names[:2]+"\t", end='')
        print()
        for i in resultado.mat_confusao:
            print(numbers[i][0:2]+"\t", end='')
            for j in resultado.mat_confusao[i]:
                print(str(resultado.mat_confusao[i][j])+"\t", end='')
            print()
        print('\n')
        print('===================================================================\n')
    
def parameters_graph(trials_fold) -> None:
    # build a graph for passed trials fold parameters swap
    data = [{**trial.params, 'loss': trial.value} for trial in trials_fold]
    hip.Experiment.from_iterable(data).display(force_full_width=True)