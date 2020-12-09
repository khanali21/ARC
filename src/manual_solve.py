#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
import solution_lib_0dfd9992
import solution_lib_045e512c
import solution_lib_ff805c23

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
    ''' Task Type: Identifying the pattern and repeating it along given axis
    Step 1: Getting all the pattern, we use a sliding window technique to create all possible matrices of 3x3 (in this case 441 matrices)
    Step 2: We then identify the pattern to repeat by just finding the most filled matrix from the above.
    Step 3: We then identify the partial patterns from the remaining patterns by going in different directions from the reference pattern
    Step 4: Finally we fill in partial patterns with the correct pattern and color and then repeat them along given axis.

    Please refere to the solution_lib_045e512c for more details on the logic.

    '''
    patterns = solution_lib_045e512c.get_all_patterns(x) 
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

def solve_ff805c23(x):
    """Task Type: Reveal the hiddens squares based on the symetrical pattern in the image.
    Although this was the simplest of three tasks but this required a tricky logic to mirror and flip the matrix squares.
    This was cleanest of all solutions. We determine the axis to mirror, it depends on which quadrant hidden squares lie.
    Then we get the mirror squares which have all the revealed squares and then finally we transpose them. 
    Some cool python np functions helped here.

    Please refere to the solution_lib_ff805c23 for more details on the logic.
    """
    (axis, side) = solution_lib_ff805c23.determine_mirror_axis(x)
    if (axis == 'v'):
        a = solution_lib_ff805c23.mirror_it_vertical(x)
    if (axis == 'h'):
        a = solution_lib_ff805c23.mirror_it_horizontal(x, side)
    result = solution_lib_ff805c23.get_hidden_squares(x, a)
    return result


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

