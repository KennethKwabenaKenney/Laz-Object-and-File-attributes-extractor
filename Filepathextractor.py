# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:34:19 2023

@author: kenneyke
"""

import os
import pandas as pd
import re
from PyQt5.QtWidgets import QApplication, QFileDialog

# Create a PyQt application instance
app = QApplication([])

# Ask the user to select a folder
folder_path = QFileDialog.getExistingDirectory(None, "Select Folder Containing Files", ".", QFileDialog.ShowDirsOnly)

# Get a list of files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.laz')]

# Extract and sort file paths based on the numbers in brackets
sorted_files = sorted(files, key=lambda x: int(x.split('(')[1].split(')')[0]) if '(' in x and ')' in x else float('inf'))

# Check if there are any valid files
if not sorted_files:
    print("No valid files found.")
else:
    # Create a DataFrame to store the file paths
    df = pd.DataFrame({'File Paths': [os.path.join(folder_path, file) for file in sorted_files]})

    # Use regular expression to capture the part before the second '('
    first_file_name = re.split(r'\[', sorted_files[0], maxsplit=1)[0] + '[' + re.split(r'\[', sorted_files[0], maxsplit=1)[1].split('[', 1)[0]
    excel_file_name = first_file_name + '.xlsx'
    excel_file_path = os.path.join(folder_path, excel_file_name)

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)

    print(f"File paths saved to {excel_file_path}")
