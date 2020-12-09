# Solution Library for problem 045e512c
import numpy as np
"""
Solutions by Muhammad Ali Khan (Student ID 20235525)
"""

def get_sub_matrix(row, column, x):
    """Return the 3x3 matrix starting from the given (row, column) square. If the matrix goes beyond dimensions of the main matrix (x)
    it fills in the squares in resultant matrix with 0
    """
    result = []
    for i in range(3):
        result_row = []
        for j in range(3):
            if (column+j) < len(x[row]) and (row+i) < len(x[row]):
                result_row.append(x[row+i][column+j])
        if len(result_row) < 3:
            for a in range(3-len(result_row)):
                result_row.append(0)
        result.append(result_row)
    return (row, column, np.array(result))


def get_all_patterns(x):
    """Using the sliding window technique, this finds out all the 3x3 matrix in the main matrix (x).
    It then returns only those matrix which has non-zero elements. (meaning have at least one colored square) 
    """
    matrices = []
    for row in range(len(x)):
        for column in range(len(x[row])):
            matrices.append(get_sub_matrix(row,column,x) )
    
    result = dict()
    for index, (row, column, sub_matrix) in enumerate(matrices):
        if not np.all(sub_matrix==0):
            result[index] = (row, column, sub_matrix)
    return result

def get_pattern_to_repeat(patterns):
    """This returns the matrix which has the most non-zero (colored squares) elements. It also returns the location of that matrix in the main.
    """
    pattern_to_repeat = None
    highest = 0
    for key, (row, column, value) in patterns.items():
        if np.count_nonzero(value) > highest:
            highest = np.count_nonzero(value)
            pattern_to_repeat = (row, column, value)   
    return pattern_to_repeat


def get_color(matrix):
    """Returns the color of the matrix (excluding black)
    """
    for a in matrix:
        for color in a:
            if color != 0:
                return color

def replace_color(matrix, color):
    """It creates a new copy of the matrix and then replaces the black squares in that matrix with the given color
    """
    result = np.copy(matrix)
    for i in range(len(result)):
        for j in range(len(result)):
            if result[i][j] != 0:
                result[i][j] = color
    return result            


def get_colored_matrix(matrix, pattern_matrix):
    """Return the matrix filled with the color for black squares. This returns the location of the matrix too.
    """
    (row, column, value) = matrix
    color = get_color(value)
    new_matrix = replace_color(pattern_matrix, color)
    return (row, column, new_matrix )


"""
The following methods implement the logic to fill in the matrix with the colored matrix along a given axis.
this is a quick and dirty implemenation and can be improved with some better logic. 
There are 8 directions namely : left, right, top, bottom, top_left, bottom_left, top_right and bottom_right.
Two of them are not implemented in the interest of time as they were not needed.
"""
def repeat_bottom_right_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(row, len(x) + 10 , 4):
        for r in range(a, a+3):
            x_column = column + (i*4)
            for c in range(x_column, x_column+3):
                sub_matrix_row = (r-row) - (i*4)
                sub_matrix_col = (c - column) - (i*4)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1

    return matrix 


def repeat_top_right_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(row, -10 , -4):
        for r in range(a, a+3):
            x_column = column + (i*4)
            for c in range(x_column, x_column+3):
                sub_matrix_row = (r-row) + (i*4)
                sub_matrix_col = (c - column) - (i*4)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1
    return matrix 
    


def repeat_left_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(column, -10, -4):
        for r in range(row, row+3):
            for c in range(a, a+3):
                sub_matrix_row = r-row
                sub_matrix_col = (c - column) + (i*4)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1
                
    return matrix

def repeat_right_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(column, len(x[0]) + 10, 4):
        for r in range(row, row+3):
            for c in range(a, a+3):
                sub_matrix_row = r-row
                sub_matrix_col = (c - column) - (i*4)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1
                
    return matrix

def repeat_up_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(row, -10 , -4):
        for r in range(a, a+3):
            for c in range(column, column+3):
                sub_matrix_row = (r-row) + (i*4)
                sub_matrix_col = (c - column)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1
    return matrix 

def repeat_down_matrices(matrix, x):
    row, column, pattern = matrix
    i = 0
    for a in range(row, len(x) + 10 , 4):
        for r in range(a, a+3):
            for c in range(column, column+3):
                sub_matrix_row = (r-row) - (i*4)
                sub_matrix_col = (c - column)
                if (r>=0 and r < len(x)) and (c >= 0 and c < len(x[r])):
                    x[r][c] = pattern[sub_matrix_row][sub_matrix_col]
        i = i + 1

    return matrix   

def repeat_bottom_left_matrices(matrix, x):
    '''
    did not implement intentionally as was needed for the current assignment. But the logic remains same
    '''
    return matrix   

def repeat_top_left_matrices(matrix, x):
    '''
    did not implement intentionally as was needed for the current assignment. But the logic remains same
    '''
    return matrix   
