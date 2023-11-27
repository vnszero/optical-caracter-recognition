import numpy as np
import pandas as pd
import hiplot as hip
from evaluation import Experiment

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

def show_results(name:str, numbers:dict, exp:Experiment) -> None:
    print(name)
    for result in exp.results:
        print("Macro f1: " + str(result.macro_f1))
        print("\nAccuracy: " + str(result.accuracy))
        print("\nf1 per category: ")
        for key in result.f1_per_category.keys():
            print('\t{} -> {}'.format(numbers[key], result.f1_per_category[key]))
        print("\nprecision: ")
        for key in result.precision.keys():
            print('\t{} -> {}'.format(numbers[key], result.precision[key]))
        print("\nrecall: ")
        for key in result.recall.keys():
            print('\t{} -> {}'.format(numbers[key], result.recall[key]))
        print('\nConfusion Matrix: ')
        print("\t", end='')
        for names in numbers.values():
            print(names[:2]+"\t", end='')
        print()
        for i in result.confusion_matrix:
            print(numbers[i][0:2]+"\t", end='')
            for j in result.confusion_matrix[i]:
                print(str(result.confusion_matrix[i][j])+"\t", end='')
            print()
        print('\n')
        print('===================================================================\n')
    
def parameters_graph(trials_fold) -> None:
    # build a graph for passed trials fold parameters swap
    data = [{**trial.params, 'loss': trial.value} for trial in trials_fold]
    hip.Experiment.from_iterable(data).display(force_full_width=True)

def info_gain_matrix_recover(info_gain_database:pd.DataFrame, square_reference:int) -> list:

    # Start
    info_gain_matrix_index = 0
    info_gain_matrix = []
    info_gain_matrix.append([])

    # Pass through the DataFrame's rows and fill matrix with values calculated
    line_breaker = 0
    for index, row in info_gain_database.iterrows():
        for item in row:
            if not('pixel' in str(item)):
                if info_gain_matrix_index < square_reference:
                    info_gain_matrix[info_gain_matrix_index].append(item)
                    line_breaker += 1
                    if (line_breaker % square_reference == 0):
                        info_gain_matrix_index += 1
                        if info_gain_matrix_index < square_reference:
                            info_gain_matrix.append([])
                        line_breaker = 0
    
    return info_gain_matrix

def create_filter_array(square_reference:int, start_row:int, start_column:int, last_row:int, last_column:int) -> list:
    filtr = np.zeros((28,28))
    for i in range(start_row, last_row):
        for j in range(start_column, last_column):
            filtr[i][j] = 1

    filtr_array = []
    for i in range(28):
        for j in range(28):
            filtr_array.append(int(filtr[i][j]))
            
    # key 2 to indicate the class
    filtr_array.append(2)
    
    return filtr_array

def create_selected_base(input_file_name:str, output_file_name:str, filtr_array:list) -> None:
    #handling file and filter by info gain 
    new_file = open(output_file_name, 'w')
    with open(input_file_name, 'r') as file:
        first_row = True
        for line in file:
            row_to_write = ''
            items = line.split(',')
            for i in range(len(items)):
                if filtr_array[i] == 1:
                    row_to_write += items[i]+','
                elif filtr_array[i] == 2:
                    row_to_write += items[i]
            nan = False
            if not first_row:
                for item in row_to_write:
                    if not item.isnumeric() and not item.isalpha() and item!=',' and item!='\n':
                        nan = True
            if row_to_write.count(',') == 400 and not nan:
                new_file.write(row_to_write)
            first_row = False

    new_file.close()

def create_binary_base(input_file_name:str, output_file_name:str) -> None:
    #handling file and transform numbers to 0 or 1
    new_file = open(output_file_name, 'w')
    with open(input_file_name, 'r') as file:
        first_row = True
        for line in file:
            row_to_write = ''
            items = line.split(',')
            for i in range(len(items)):
                if first_row:
                    if '\n' in items[i]:
                        row_to_write += items[i]
                    else:
                        row_to_write += items[i]+','
                else:
                    if '\n' in items[i]:
                        row_to_write += items[i]
                    else:
                        if float(items[i]) == 0.0:
                            row_to_write += '0,'
                        else:
                            row_to_write += '1,'
            new_file.write(row_to_write)
            first_row = False

    new_file.close()