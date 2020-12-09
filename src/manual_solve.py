#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

"""
    Solutions by Muhammad Ali Khan (Student ID 20235525)
    Github Forked Repo URL : https://github.com/khanali21/ARC

    Introduction: 
    This code contains solutions to three tasks which require some unique transformation actions which have been 
    implemented in the solution_lib_* for each individual task. Below is the brief discription of each task and 
    the transformation required to solve each of them. The implementation logic is described in each of the solution_lib
    file please refer to them.
    * solution_lib_0dfd9992
    * solution_lib_045e512c
    * solution_lib_ff805c23
    (For the submission these are now part of this file)

    Task: 0dfd9992.json
    Although this task is a trivial task of identifying the hidden repeate patterns based on the given patterns in the full matrix,
    yet in my opinion it is key to the understanding of match and merge problems. Initially I had approached it to identify hidden
    pattern underneath the black squares, but there was a very neat trick to just identify the row that closely matches with the row 
    which has black squares and fill the squares with colors that identify from the matching row. This process needed to be done repeatedly
    until all hidden squares are filled. 

    Task: 045e512c.json
    This is another tasks of one of the common kinds. It is a simple repeated pattern task which requires identification of embedded pattern.
    And once the pattern is identified we need to repeat the same pattern in the direction given by squares in the input. This tasks included 
    two key problems, identifying the pattern, that require a sliding window technique to slide a 3x3 matrix across the main matrix and
    pick the matrix which has most colored squares. 
    Second problem was to identify the partial patterns and their direction which could left, right, up, down, top_left, top_right, bottom_left
    and bottom_right. Finally we needed to just repeat the pattern in the given direction with the correct color.

    Task: ff805c23.json
    This problem has some resemblance with the 0dfd9992, but it has a different transformation logic. Essentially we need to identify the 
    mirror pattern, in the symmetry of the matrix. Although this appears to be a trivial problem, however it requires a clean logic to identify
    the mirror part which has the revealed squares and then being able to fill in those squares. I used a bit python tricks here to mirror 
    and transpose and merge the halves.


    Full Disclosure & Credit: This was a very interesting assignment, I took the help of my 14 years old son (Hassan Ali Khan) to identify 
    three unique diffecult problems. In the process he solved many of those problems. 


    Reflections on Chollet paper:
    The definition of "inteligence" both in philosophical and AI context has been an interesting topic for me. Chollet's paper provides a 
    very insightful commentary on the dual definition of intelligence in the current contex of AI research. Back in 96, when I took the course
    of "Knowledge Based Systems" my and probably most of the students interest was developing the best chess playing program. But as we have 
    seen over the last couple of decades that both "crystallized skill" as in implementing a chess program and "skill-acquisition ability"
    (as in modern AI applications) are the foundation for understanding the intelligence. 
     
     
"""

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


def solve_0dfd9992(x):
    '''A simple task to identify hidden squares based on repeated pattern in the main matrix.
    The code uses a simple match and merge technique. We first identify all the rows which have hidden squares (color=0)
    After that we find the row that exactly matches the square colors in the given row (excluding the hidden squares).
    We merge the known colors for those two rows and create our new row.
    We repeat this process 10 times or until all the squares have been filled with correct colors.
    Please refere to the solution_lib_0dfd9992 for more details on the logic.
    '''
    rows = rows_with_missing_squares(x)
    new_rows = dict()
    for row in rows:
        new_row = match_and_merge(x[row], x)
        i = 0
        while ( (np.count_nonzero(new_row) != len(new_row)) and i < 10):
            i = i + 1
            new_row = match_and_merge(new_row, x)
            
        new_rows[row] = match_and_merge(x[row], x)

    for row in rows:
        if row in new_rows.keys(): 
            x[row] =  new_rows[row]
        
    return x

def solve_045e512c(x):
    ''' Task Type: Identifying the pattern and repeating it along given axis
    Step 1: Getting all the pattern, we use a sliding window technique to create all possible matrices of 3x3 (in this case 441 matrices)
    Step 2: We then identify the pattern to repeat by just finding the most filled matrix from the above.
    Step 3: We then identify the partial patterns from the remaining patterns by going in different directions from the reference pattern
    Step 4: Finally we fill in partial patterns with the correct pattern and color and then repeat them along given axis.

    Please refere to the solution_lib_045e512c for more details on the logic.

    '''
    patterns = get_all_patterns(x) 
    pattern_to_repeat = get_pattern_to_repeat(patterns)
    p_row, p_column, p_value = pattern_to_repeat
    left = None
    right = None
    up = None
    down = None
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    for key, (row, column, value) in patterns.items():
        if (row == (p_row+4) and column == p_column):
            down = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column):
            up = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if column == (p_column-4) and row == p_row:
            left = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (column == (p_column+4) and row == p_row):
            right = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column+4):
            top_right = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row+4) and column == p_column+4):
            bottom_right = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column-4):
            top_left = get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row+4) and column == p_column-4):
            bottom_left = get_colored_matrix((row, column, value), pattern_to_repeat[2])

    if left:
        repeat_left_matrices(left, x)
    if right:
        repeat_right_matrices(right, x)
    if up:
        repeat_up_matrices(up, x)
    if down: 
        repeat_down_matrices(down,x)
    if top_right:
        repeat_top_right_matrices(top_right, x)
    if  bottom_right:       
        repeat_bottom_right_matrices(bottom_right,x)
    if top_left:        
        repeat_top_left_matrices(top_left, x)
    if  bottom_left:       
        repeat_bottom_left_matrices(bottom_left, x)
    
    return x

def solve_ff805c23(x):
    """Task Type: Reveal the hiddens squares based on the symetrical pattern in the image.
    Although this was the simplest of three tasks but this required a tricky logic to mirror and flip the matrix squares.
    This was cleanest of all solutions. We determine the axis to mirror, it depends on which quadrant hidden squares lie.
    Then we get the mirror squares which have all the revealed squares and then finally we transpose them. 
    Some cool python np functions helped here.

    Please refere to the solution_lib_ff805c23 for more details on the logic.
    """
    (axis, side) = determine_mirror_axis(x)
    if (axis == 'v'):
        a = mirror_it_vertical(x)
    if (axis == 'h'):
        a = mirror_it_horizontal(x, side)
    result = get_hidden_squares(x, a)
    return result

"""
Following section contains the 3 solution libs.
"""
# Solution Library for problem 0dfd9992
"""
Solutions by Muhammad Ali Khan (Student ID 20235525)
"""

def match(a1, a2):
    '''This is simple matching algorithm to match two lists. It ignores the columns with '0' in it and matches only the elements that have non-zero values in both lists.
    '''
    result = [None] * len(a1)
    for i, (x,y) in enumerate(zip(a1, a2)):
        if x != 0 and y != 0 and x!=y:
            result[i] = -1
            continue
        if x == 0 or y == 0:
            result[i] = 0
            continue
        if x != 0 and y != 0 and x == y:
            result[i] = 1
            continue
        result[i] = 0
    matched = True
    for x in result:
        if x == -1:
            matched = False
    return matched 

def merge(a1, a2):
    """This is simple merging algorithm to merge two lists. It merges the two lists such that 
    the element with value zero in one list are replaced by the corresponding non-zero element from the other list.
    """
    result = [None] * len(a1)
    for i, (x,y) in enumerate(zip(a1, a2)):
        if x != 0 and y != 0 and x==y:
            result[i] = x
            continue
        if x == 0 and y != 0:
            result[i] = y
            continue
        if x != 0 and y == 0:
            result[i] = x
            continue
        result[i] = x
    return result            

def match_and_merge(actual_row, input):
    """This returns the final row after matching a corresponding row in the input that closely matches
    """
    if np.count_nonzero(actual_row) == len(actual_row):
        return actual_row
    matching_rows = []
    for tocompare in input:
        if match(actual_row, tocompare): 
            matching_rows.append(tocompare)
    x = actual_row
    for tocompare in matching_rows:
        if np.all(actual_row==tocompare): # Ignore if the same column
            continue      
        if match(actual_row, tocompare):   
            x =  merge(x, tocompare)    
    return x

def rows_with_missing_squares(squares):
    """This searches for the squares with 0 value and returns the list of corresponding rows.

    """
    rows = []
    for i, square in enumerate(squares):
        if 0 in square:
            rows.append(i)
    return list(set(rows))        


# Solution Library for problem 045e512c
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

# Solution Library for problem ff805c23
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




def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

