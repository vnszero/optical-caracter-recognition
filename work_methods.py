import numpy as np
import pandas as pd

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