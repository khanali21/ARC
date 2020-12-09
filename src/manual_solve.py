#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
import solution_lib_0dfd9992
import solution_lib_045e512c

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_0dfd9992(x):
    '''A simple pattern to remove unused (black) squares and merge the colored squares to form smaller grid.

    '''
    rows = solution_lib_0dfd9992.rows_with_missing_squares(x)
    new_rows = dict()
    for row in rows:
        new_row = solution_lib_0dfd9992.match_and_merge(x[row], x)
        i = 0
        while ( (np.count_nonzero(new_row) != len(new_row)) and i < 10):
            i = i + 1
            new_row = solution_lib_0dfd9992.match_and_merge(new_row, x)
            
        new_rows[row] = solution_lib_0dfd9992.match_and_merge(x[row], x)

    for row in rows:
        if row in new_rows.keys(): 
            x[row] =  new_rows[row]
        
    return x

def solve_045e512c(x):
    ''' Task Type: Identifying the pattern and filling the correct color
        A simple pattern to identify the grid based on the color and then fill that grid with one color in a corresponding squares. 
    '''
    patterns = solution_lib_045e512c.identify_pattern(x) 
    pattern_to_repeat = solution_lib_045e512c.get_pattern_to_repeat(patterns)
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
            down = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column):
            up = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if column == (p_column-4) and row == p_row:
            left = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (column == (p_column+4) and row == p_row):
            right = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column+4):
            top_right = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row+4) and column == p_column+4):
            bottom_right = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row-4) and column == p_column-4):
            top_left = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])
        if (row == (p_row+4) and column == p_column-4):
            bottom_left = solution_lib_045e512c.get_colored_matrix((row, column, value), pattern_to_repeat[2])

    if left:
        solution_lib_045e512c.repeat_left_matrices(left, x)
    if right:
        solution_lib_045e512c.repeat_right_matrices(right, x)
    if up:
        solution_lib_045e512c.repeat_up_matrices(up, x)
    if down: 
        solution_lib_045e512c.repeat_down_matrices(down,x)
    if top_right:
        solution_lib_045e512c.repeat_top_right_matrices(top_right, x)
    if  bottom_right:       
        solution_lib_045e512c.repeat_bottom_right_matrices(bottom_right,x)
    if top_left:        
        solution_lib_045e512c.repeat_top_left_matrices(top_left, x)
    if  bottom_left:       
        solution_lib_045e512c.repeat_bottom_left_matrices(bottom_left, x)
    
    return x



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

