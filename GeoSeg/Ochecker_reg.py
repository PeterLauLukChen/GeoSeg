import CandidateSearching as cs
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import linregress

"""
This script analyzes overlapping regions of specific cell values across different layers. 
It applies linear and quadratic regression models to assess the continuity and trend of 
overlapping area across cell masks.

Key functions:
1. linear_r_squared(): Computes the R² value for a linear fit.
2. quadratic_r_squared(): Computes the R² value for a quadratic fit.
3. O_index_visualizor(): Evaluates whether a cell satisfies an overlap threshold.
4. O_index_result(): Computes and returns the best-fitting regression metric.
5. O_checker(): Identifies increasing, decreasing, or turning point patterns in data.
6. main(): For checking how many cell satisfying the linear or quadratic criterion
"""

"""
Different attributes we need to store for a cell

Parameters:
value: (int) the value of cell in 2D segmentation numpy
highest_layer: (int) the highest layer of the cell 
lowest_layer: (int) the lowest layer of the cell
"""
class Cell:
    def __init__(self, value, highest_layer, lowest_layer):
        self.value = value
        self.highest_layer = highest_layer
        self.lowest_layer = lowest_layer

"""
Computes the R² value for a linear regression model fitted to the given array.

Parameters:
arr (list of float): Sequence of values representing overlap areas.

Returns:
float: R² value 
"""
def linear_r_squared(arr):
    if np.isnan(arr).any() or len(arr) < 2:
        return np.nan
    x = np.arange(len(arr))
    slope, intercept, r_value, p_value, std_err = linregress(x, arr)
    return r_value ** 2 if not np.isnan(r_value) else np.nan

"""
Computes the R² value for a quadratic regression model fitted to the given array.

Parameters:
arr (list of float): Sequence of values representing overlap areas.

Returns:
float: R² value indicating the goodness of fit for a quadratic model.
"""
def quadratic_r_squared(arr):
    if np.isnan(arr).any() or len(arr) <= 2:
        return 1
    x = np.arange(len(arr))
    coeffs = np.polyfit(x, arr, 2)
    predicted = np.polyval(coeffs, x)
    total_sum_squares = np.sum((arr - np.mean(arr))**2)
    residual_sum_squares = np.sum((arr - predicted)**2)
    r_squared = 1 - (residual_sum_squares / (total_sum_squares + 0.0001))
    return r_squared

"""
Evaluates if a cell satisfies overlap criteria using regression models.

Parameters:
Cell_val (int): The cell value identifier.
highest_layer (int): The highest layer containing the cell.
lowest_layer (int): The lowest layer containing the cell.
array (3D array): Layered representation of cell distributions.
sigma (float): Threshold for significance based on standard deviation.

Returns:
int: 1 if criteria are met, 0 otherwise.
"""
def O_index_visualizor(Cell_val, highest_layer, lowest_layer, array, sigma):
    result = []

    for layer in range(highest_layer, lowest_layer):
        cellA_coordinates = set()
        for row_idx, row in enumerate(array[layer]):
            for col_idx, cell_value in enumerate(row):
                if cell_value == Cell_val:
                    cellA_coordinates.add((row_idx, col_idx))
        
        cellB_coordinates = set()

        for row_idx, row in enumerate(array[layer + 1]):
            for col_idx, cell_value in enumerate(row):
                if cell_value == Cell_val:
                    cellB_coordinates.add((row_idx, col_idx))

        Overlapping_area = len(cellA_coordinates.intersection(cellB_coordinates))
        result.append((layer, Overlapping_area))

    y_values = [point[1] for point in result]   
    O_checker_result = O_checker(y_values)

    if O_checker_result == 1:
        return 1
    
    elif O_checker_result == -1:
        return 1
    
    elif O_checker_result == 0 and sigma != 0:
        linear_result = linear_r_squared(y_values)

        # Statistics from plant dataset using for pretraining 
        linear_mean = 0.9377
        linear_std = 0.0715

        if linear_result >= linear_mean -  sigma * linear_std:
            return 1
        
        quadratic_result = quadratic_r_squared(y_values)

        # Statistics from plant dataset using for pretraining
        quadratic_mean = 0.9149
        quadratic_std = 0.1109

        if quadratic_result >= quadratic_mean - sigma * quadratic_std:
            return 1
        
        return 0
    
    else:
        return 0

"""
Determines whether linear or quadratic regression better explains overlap variation.

Parameters:
Cell_val (int): The cell value identifier.
highest_layer (int): The highest layer containing the cell.
lowest_layer (int): The lowest layer containing the cell.
array (3D array): Layered representation of cell distributions.

Returns:
tuple: (Best R² value, 0 for linear or 1 for quadratic).
"""    
def O_index_result(Cell_val, highest_layer, lowest_layer, array):
    result = []
    for layer in range(highest_layer, lowest_layer):
        cellA_coordinates = set()
        for row_idx, row in enumerate(array[layer]):
            for col_idx, cell_value in enumerate(row):
                if cell_value == Cell_val:
                    cellA_coordinates.add((row_idx, col_idx))
        
        cellB_coordinates = set()
        for row_idx, row in enumerate(array[layer + 1]):
            for col_idx, cell_value in enumerate(row):
                if cell_value == Cell_val:
                    cellB_coordinates.add((row_idx, col_idx))

        Overlapping_area = len(cellA_coordinates.intersection(cellB_coordinates))
        result.append((layer, Overlapping_area))

    y_values = [point[1] for point in result]
    O_checker_result = O_checker(y_values)

    if O_checker_result == 2:
        return 1, 0
    
    elif O_checker_result == 3:
        return 1, 1
    
    else:
        linear_result = linear_r_squared(y_values)
        quadratic_result = quadratic_r_squared(y_values)

        if linear_result > quadratic_result:
            return linear_result, 0
        
        else:
            return quadratic_result, 1
"""
Checks the trend of the given sequence to determine its pattern.

Parameters:
arr (list of float): Sequence of overlap values.

Returns:
int: 
    1 for strictly increasing or decreasing,
    -1 for a single turning point,
    0 for violating both.
"""
def O_checker(arr):
    increasing = decreasing = True
    turning_point_found = False

    for i in range(1, len(arr)):
        if arr[i] > arr[i - 1]:
            decreasing = False

        elif arr[i] < arr[i - 1]:
            if increasing:
                turning_point_found = True

            increasing = False

        else:
            return 0
        
    if increasing or decreasing:
        return 1
    
    elif turning_point_found:
        return -1

"""
For checking how many cell satisfying the linear or quadratic criterion; 
Not necessaily a module for GeoSeg.

Parameters:
ranges (int): Number of files to process.
dir (str): Directory containing dataset files.
file (str): File naming pattern.
sigma (float): Threshold parameter for regression evaluation.
"""
def main(ranges,dir,file,sigma):
    directory = dir
    cnt = 0
    passed = 0

    for i in range(ranges):
        file_name = file.format(i)
        file_path = os.path.join(directory, file_name)

        if os.path.exists(file_path):
            array = cs.load_array(file_path)
            cell_dict = cs.extract_cells_info(array)

            for cell_val, cell_info in cell_dict.items():
                cnt += 1
                passed += O_index_visualizor(cell_val, cell_info.highest_layer, cell_info.lowest_layer, array, sigma)

    print(f"Number of total cell: {cnt}")
    print(f"Number of total cells that passed all the strict criterion: {passed}")
    print(f"Accuracy: ",passed/cnt)

if __name__ == "__main__":
    main()