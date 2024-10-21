# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:50:54 2023

@author: kenneyke
"""

import os
import pandas as pd
import re
from PyQt5.QtWidgets import QApplication, QFileDialog
import laspy
import numpy as np

#%% Function definitions

# Function for reading point cloud
def readPtcloud(file_path):
    L = laspy.read(file_path)
    ptcloud = np.array((L.x, L.y, L.z, L.Ext_Class)).transpose()
    return ptcloud

#Function for computing bounding box
def compute_bounding_box(point_cloud):
    # Initialize min and max values with the first point
    min_values = list(point_cloud[0])
    max_values = list(point_cloud[0])

    # Iterate through the point cloud to compute extremes
    for point in point_cloud[1:]:
        min_values = [min(p, q) for p, q in zip(min_values, point)]
        max_values = [max(p, q) for p, q in zip(max_values, point)]
    
    """    
    # Create the Bounding Box Vertices
    bounding_box_vertices = [
        (min_values[0], min_values[1], min_values[2]),  # min (X), min (Y), min (Z)
        (max_values[0], min_values[1], min_values[2]),  # max (X), min (Y), min (Z)
        (min_values[0], max_values[1], min_values[2]),  # min (X), max (Y), min (Z)
        (max_values[0], max_values[1], max_values[2])   # max (X), max (Y), max (Z)
    ]
    """
    # Calculate Dimensions
    dimension_X = [max_values[0] - min_values[0]]  # Width along X-axis
    dimension_Y = [max_values[1] - min_values[1]]  # Width along Y-axis
    dimension_Z = [max_values[2] - min_values[2]]  # Depth along Z-axis

    return dimension_X, dimension_Y, dimension_Z

#%%

# Create a PyQt application instance
app = QApplication([])

# Ask the user to select a folder
folder_path = QFileDialog.getExistingDirectory(None, "Select Folder Containing Files", ".", QFileDialog.ShowDirsOnly)

# Get a list of files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.laz')]

# Extract and sort file paths based on the numbers in brackets
sorted_files = sorted([os.path.join(folder_path, file) for file in files],
                      key=lambda x: int(x.split('(')[1].split(')')[0]) if '(' in x and ')' in x else float('inf'))

# Check if there are any valid files
if not sorted_files:
    print("No valid files found.")
else:
    
    # Create a DataFrame to store the file paths
    df = pd.DataFrame({'File Paths': sorted_files})
    
    # Create a DataFrame to store the file paths and bounding box dimensions
    data = {'File Paths': sorted_files, 'Bound_Box Dim_X': [], 'Bound_Box Dim_Y': [], 'Bound_Box Dim_Z': []}

    for file_path in sorted_files:
        # Read point cloud from LAS file
        point_cloud = readPtcloud(file_path)

        # Compute bounding box dimensions
        dimX, dimY, dimZ = compute_bounding_box(point_cloud)

        # Add dimensions to the list
        data['Bound_Box Dim_X'].append(dimX)
        data['Bound_Box Dim_Y'].append(dimY)
        data['Bound_Box Dim_Z'].append(dimZ)

    # Create a DataFrame from the dictionary
    df_dimensions = pd.DataFrame(data)
    
    # Extract the folder name from the folder_path
    folder_name = os.path.basename(folder_path)
    
    # Get the parent directory (one step outside the selected folder)
    parent_directory = os.path.dirname(folder_path)
    
    # Generate the Excel file name with the folder name
    excel_file_name = f"{folder_name}_data.xlsx"

    # Combine the parent directory and file name
    excel_file_path = os.path.join(parent_directory, excel_file_name)

    # Merge the two DataFrames on the 'File Paths' column
    df_merged = pd.merge(df, df_dimensions, on='File Paths')

    # Remove square brackets from 'Bound_Box Dimensions' column
    df_merged['Bound_Box Dim_X'] = df_merged['Bound_Box Dim_X'].apply(lambda x: x[0])
    df_merged['Bound_Box Dim_Y'] = df_merged['Bound_Box Dim_Y'].apply(lambda x: x[0])
    df_merged['Bound_Box Dim_Z'] = df_merged['Bound_Box Dim_Z'].apply(lambda x: x[0])

    # Save the merged DataFrame to an Excel file
    df_merged.to_excel(excel_file_path, index=False)

    print(f"File paths and bounding box dimensions saved to {excel_file_path}")