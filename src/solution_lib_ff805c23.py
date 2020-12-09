# Solution Library for problem ff805c23
import numpy as np
"""
Solutions by Muhammad Ali Khan (Student ID 20235525)
"""


def get_hidden_matrix_dimension(x):
    """This searches for the matrix which is hiding the squares, the hidden square color is 1 (blue).
    It returns the location of the matrix (start_row, start_column, end_row, end_column)
    """
    hidden_square_color = 1
    start_row = -1
    start_column = -1
    dimenson_x = 0
    dimenson_y = 0

    for row in range(len(x)):
        for col in range(len(x[row])):
            if x[row][col] == hidden_square_color:
               if start_row == -1:
                   start_row = row
                   start_column = col
               if row == start_row:
                dimenson_x = dimenson_x + 1    

    end_column = -1
    end_row = -1
    for row in range(len(np.transpose(x))):
        for col in range(len(x[row])):
            if x[row][col] == hidden_square_color:
               if end_column == -1:
                   end_column = row
                   end_row = col
               if row == start_row:
                dimenson_y = dimenson_y + 1 
    end_row = start_row + dimenson_y
    end_column = start_column + dimenson_x
    return (start_row, start_column, end_row, end_column)

def determine_mirror_axis(x):
    """This determines the axis of symmetry depending upon the location of the hidden matrix. 
    If the hidden matrix is crossing the mid row on left or right we have vertical symmetry.
    If the hidden matrix is crossing the mid column on top or bottom we have horizontal symmetry
    else we return horizontal. 
    It also return the side which has the revealed squars.
    """
    start_row, start_column, end_row, end_column = get_hidden_matrix_dimension(x)
    mid_row = int(len(x)/2)
    mid_col = int(len(x[0])/2)
    side = 'up'
    axis = 'h'
    if start_row > mid_row:
        side = 'down'
    if mid_row > start_row and mid_row < end_row:
        axis = 'v'
    if mid_col > start_column and mid_col < end_column:
        axis = 'h'
    return (axis, side)    

def mirror_it_vertical(x):
    """
    Transpose the matrix, divide it into two and then reverse it (flip), and then merge and transpose agian, return the resultant matrix
    """
    t_x = np.transpose(x)
    dimension_y = len(x)
    half1 = []
    half2 = []
    for row in range(int(dimension_y/2)):
        half1.append(t_x[row])
    for row in range( int(dimension_y/2), len(t_x), 1):
        half2.append(t_x[row])
    mirror = list(reversed(half2))
    a = mirror + half2
    b = np.transpose(a)
    return b

def mirror_it_horizontal(x, side):
    """
        Divide the matrix, depending upon the side (up or down), reverse the half which is has revealed squares and then merge the two halfs.
    """
    dimension_y = len(x)
    half1 = []
    half2 = []
    for row in range(int(dimension_y/2)):
        half1.append(x[row])
    for row in range( int(dimension_y/2), len(x), 1):
        half2.append(x[row])
    if (side == 'up'):
        mirror = list(reversed(half2))
        a = mirror + half2
    else:
        mirror = list(reversed(half1))    
        a = half1 + mirror

    return a

def get_hidden_squares(x, a):
    """
    Return the matrix consisting of the hidden squares.
    """
    hidden_square_color = 1
    output_rectangle = []
    start_row = -1
    start_column = -1
    dimenson_x = 0
    dimenson_y = 0

    for row in range(len(x)):
        for col in range(len(x[row])):
            if x[row][col] == hidden_square_color:
               if start_row == -1:
                   start_row = row
                   start_column = col
               if row == start_row:
                dimenson_x = dimenson_x + 1    

    end_column = -1
    end_row = -1
    for row in range(len(np.transpose(x))):
        for col in range(len(x[row])):
            if x[row][col] == hidden_square_color:
               if end_column == -1:
                   end_column = row
                   end_row = col
               if row == start_row:
                dimenson_y = dimenson_y + 1 
    end_row = start_row + dimenson_y
    end_column = start_column + dimenson_x
    for i in range(start_row, end_row):
        new_row = []
        for j in range(start_column, end_column):
            new_row.append(a[i][j])
        output_rectangle.append(new_row)
    return output_rectangle

