# Solution Library for problem 0dfd9992
import numpy as np

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
    """This returns the final row after matching a correspondin row in the input that closely matches
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
    rows = []
    for i, square in enumerate(squares):
        if 0 in square:
            rows.append(i)
    return list(set(rows))        


