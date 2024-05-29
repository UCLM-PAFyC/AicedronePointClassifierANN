"""
Scalar field operations
Alberto Morcillo Sanz - TIDOP
"""

import sys

import numpy as np

import laspy


def contained(value: any, array: list[any]) -> bool:
    """
    :param value: the value
    :param: the list
    :return: returns wheter a value is inside of a list or not
    """
    for v in array:
        if v == value:
            return True
    return False


def normalize_scalar_fields(las: laspy.LasData, label:str, data: list[list[float]], ignored_scalar_fields: list[str], callback=None) -> None:
    """
    :param las: Opened .LAS file
    :data: List of features for each point [f1, f2, ..., fn], [g1, g2, ..., gn], ...., [z1, z2, ..., zn]
    :label: Label of the class
    Normalize all the scalar fields except the ignored ones and the label class and save it in data
    """
    num_points: int = len(las.points)
    
    min_max_length: int = len(list(las.point_format.dimension_names))
    min_values: np.array = np.empty(min_max_length)
    max_values: np.array = np.empty(min_max_length)
    
    min_values.fill(sys.float_info.max)
    max_values.fill(sys.float_info.min)
    
    scalar_fields = list(las.point_format.dimension_names)
    label_index = None
    
    previous_percentage: int = -1

    # Load min and max values
    for i in range(0, num_points):
        for j in range(0, len(scalar_fields)):
            
            scalar = scalar_fields[j]
            
            if contained(scalar, ignored_scalar_fields):
                continue
            
            if label is not None and scalar == label:
                label_index = j
                continue
            
            value = las[scalar][i]
            
            if value < min_values[j]:
                min_values[j] = value
            if value > max_values[j]:
                max_values[j] = value
                
        if callback is not None:
            percentage = max(int(50 * float(i) / num_points) - 1, 0)
            if percentage != previous_percentage:
                callback(percentage, normalize_scalar_fields)
                previous_percentage = percentage
    
    if callback is not None:
        callback(49, normalize_scalar_fields)
    
    # Normalize point cloud
    for i in range(0, num_points):
        
        normalized_data: list[float] = []
        
        scalar_index: int = 0
        for j in range(0, len(scalar_fields)):
            
            # IGNORE SCALAR FIELDS
            scalar = scalar_fields[j]
            if contained(scalar, ignored_scalar_fields):
                continue
            
            # DO NOT NORMALIZE LABEL
            if label is not None and scalar == label:
                label_index = scalar_index
                normalized_data.append(las[scalar][i])
                continue
            
            # Normalize current scalar field
            value = las[scalar][i]
            normalized_value = (value + abs(min_values[j])) / (max_values[j] + abs(min_values[j]))
            normalized_data.append(normalized_value)
            
            scalar_index += 1
            
        if callback is not None:
            percentage = max(int(50 * float(i) / num_points) - 1, 0) + 50
            if percentage != previous_percentage:
                callback(percentage, normalize_scalar_fields)
                previous_percentage = percentage
        
        data.append(normalized_data)
        
    # Move label to the last place
    if label is not None:
        for column in data:
            label = column.pop(label_index)
            column.append(label)
    
    if callback is not None:
        callback(100, normalize_scalar_fields)