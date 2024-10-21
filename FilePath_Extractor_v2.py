import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QFileDialog
import laspy
import numpy as np

#%% Parameters
min_dim = 0.1
n_slices = 5    # number of slices
default_path = r"D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\5_ind_objects"
    
#%% Function Definitions

# Function for reading point cloud
def readPtcloud(file_path):
    L = laspy.read(file_path)
    ptcloud = np.array((L.x, L.y, L.z, L.Ext_Class)).transpose()
    return ptcloud

# Function for computing bounding box
def compute_bounding_box(point_cloud):
    min_values = list(point_cloud[0])
    max_values = list(point_cloud[0])

    for point in point_cloud[1:]:
        min_values = [min(p, q) for p, q in zip(min_values, point)]
        max_values = [max(p, q) for p, q in zip(max_values, point)]

    dimension_X = [max_values[0] - min_values[0]]
    dimension_Y = [max_values[1] - min_values[1]]
    dimension_Z = [max_values[2] - min_values[2]]

    return dimension_X, dimension_Y, dimension_Z

# Function to calculate the majority "Ext_Class" value and its percentage
def calculate_majority_ext_class(point_cloud):
    #if not point_cloud or not all(isinstance(point, (list, tuple)) and len(point) >= 4 for point in point_cloud):
        # Check if point_cloud is not empty and all points have at least 4 elements
        #raise ValueError("Invalid point cloud structure")
    ext_class_counts = {}
    total_points = len(point_cloud)

    for point in point_cloud:
        if len(point) < 4:
            raise ValueError("Invalid point structure for one of the points")

        ext_class = point[3]
        ext_class_counts[ext_class] = ext_class_counts.get(ext_class, 0) + 1

    # Find the majority Ext_Class and its percentage
    majority_ext_class, majority_count = max(ext_class_counts.items(), key=lambda x: x[1])
    majority_percentage = (majority_count / total_points) * 100
    root_class = majority_ext_class
    # Calculate the corresponding Root_Ext_Class (root_class)
    #root_class = [int(np.floor(value / 10000)) for value in majority_ext_class]
    #root_class = int(np.floor(majority_ext_class / 10000))
    #root_class = (majority_ext_class / 10000) * 10000
    #root_class = int(root_class/10000)

    return majority_ext_class, majority_percentage, root_class

# Function to extract the "Ext_Class" values
def extract_summarized_values(point_cloud):
    summarized_values = set()  # Using a set to ensure unique values
    total_points = 0
    
    for point in point_cloud:
        if len(point) < 2:
            raise ValueError("Invalid point structure for one of the points")

        ext_class = point[3]
        
        total_points += 1

        # Add the Ext_Class value to the set
        summarized_values.add(ext_class)
        
    count = len(summarized_values)

    return summarized_values, count, total_points


# Function to add labels based on the Ext_Class values
def add_labels(ext_class_values, labels_df):
    labels_mapping = dict(zip(labels_df['Ext_Class'], labels_df['Labels']))
    return [labels_mapping[ext_class] if ext_class in labels_mapping else 'Unknown' for ext_class in ext_class_values]

'''
'''

def xyz_ptc(ptcloud):
    return ptcloud[:, :3]

# Function to calculate the center of a bounding box given its minimum and maximum coordinates
def boundingbox_center(min_xyz, max_xyz):
    bb_center_x, bb_center_y, bb_center_z = (min_xyz + max_xyz) * 0.5
    return bb_center_x, bb_center_y, bb_center_z

# Function to compute the dimension of a bounding box given its minimum and maximum coordinates
def compute_dimension(min_xyz, max_xyz):
    return max_xyz - min_xyz

# Function to compute the minimum and maximum coordinates of a bounding box for a 3D point cloud
def compute_bounding_box_new(point_cloud):
    min_xyz = np.min(point_cloud, axis=0)
    max_xyz = np.max(point_cloud, axis=0)
    return min_xyz, max_xyz

# Function to compute the dimension of the PCA (Principal Component Analysis) bounding box
def compute_pca_bounding_box(point_cloud):
    pca_point_cloud = pca_rotate_2d(point_cloud)
    pca_2d_dimx, pca_2d_dimy, pca_2d_dimz = compute_dimension(np.min(pca_point_cloud, axis=0), np.max(pca_point_cloud, axis=0))
    return pca_2d_dimx, pca_2d_dimy, pca_2d_dimz

# Function to perform a 2D PCA rotation on a 3D point cloud
def pca_rotate_2d(point_cloud):
    point_cloud_2d = point_cloud_xy(point_cloud, 0)
    pca_point_cloud = pca_rotate_3d(point_cloud_2d)
    pca_point_cloud[:, 2] = point_cloud[:, 2]
    return pca_point_cloud

# Function to perform a 3D PCA rotation on a 2D point cloud
def pca_rotate_3d(point_cloud):
    eigen_vec_0, eigen_vec_1, eigen_vec_2 = pca_eigen_vec(point_cloud)
    avg_x, avg_y, avg_z = avg_coord(point_cloud)
    pca_point_cloud = np.zeros_like(point_cloud)
    for i in range(len(point_cloud)):
        pca_point_cloud[i, 0] = np.dot(point_cloud[i] - [avg_x, avg_y, avg_z], eigen_vec_0)
        pca_point_cloud[i, 1] = np.dot(point_cloud[i] - [avg_x, avg_y, avg_z], eigen_vec_1)
        pca_point_cloud[i, 2] = np.dot(point_cloud[i] - [avg_x, avg_y, avg_z], eigen_vec_2)
    return pca_point_cloud

# Function to calculate the eigenvalues and eigenvectors for PCA
def pca_eigen_vec(point_cloud):
    cov_matrix = np.cov(point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigen_vec_0, eigen_vec_1, eigen_vec_2 = eigenvectors[:, idx]
    if eigen_vec_2[2] < 0:
        eigen_vec_2 = -eigen_vec_2  # Ensure eigen_vec_2 is positive in z
    return eigen_vec_0, eigen_vec_1, eigen_vec_2

# Function to set the z-coordinate of a 3D point cloud to a specified value
def point_cloud_xy(point_cloud, z_value=0):
    point_cloud_cpy = np.copy(point_cloud)
    point_cloud_cpy[:, 2] = z_value
    return point_cloud_cpy

# Function to calculate the average coordinates of a point cloud
def avg_coord(point_cloud):
    return np.mean(point_cloud, axis=0)

# Function to slice the 3D point cloud along the z axis
def point_cloud_3D_slice(point_cloud, n_bins):
    z_coordinates = point_cloud[:, 2]   # Extract z coordinates from the point cloud
    # Normalize the z coordinates
    normalized_z = (z_coordinates - np.min(z_coordinates)) / (np.max(z_coordinates) - np.min(z_coordinates))

    # Calculate the cut-off z values and store them into an array
    bin_edges = np.linspace(0, 1, n_bins + 1)

    # Split into slices based on the z values
    slices = []
    for i in range(n_bins):
        slice_mask = (normalized_z >= bin_edges[i]) & (normalized_z < bin_edges[i + 1])
        current_slice = point_cloud[slice_mask]
        slices.append(current_slice)
    return slices

def point_cloud_3D_slice_pca(point_cloud, n_bins):
    pca_point_cloud = pca_rotate_2d(point_cloud)
    return point_cloud_3D_slice(pca_point_cloud, n_bins)

# Function to compute the histogram of the number of points in each XY slice
def hist_npts_xy(xyz_slices):
    hist_npts = [len(slice) for slice in xyz_slices]
    return hist_npts

# Function to compute slice dimensions
def compute_slice_dimensions_3D(point_cloud_slice):

    if len(point_cloud_slice) < 1:
        dim_x, dim_y, pca_bb_diag, pca_bb_area = 0, 0, 0, 0
    elif len(point_cloud_slice) < 2:     
        dim_x, dim_y, pca_bb_diag, pca_bb_area = min_dim, min_dim, min_dim, min_dim * min_dim * np.pi * 0.25
    else:
        # Compute bounding box dimensions for the slice
        min_xy, max_xy = compute_bounding_box_new(point_cloud_slice)
        dim_x, dim_y, dim_z = compute_dimension(min_xy, max_xy)
        
        dim_x = max(dim_x, min_dim)
        dim_y = max(dim_y, min_dim)
        dim_z = max(dim_z, min_dim)
    
        # Compute PCA bounding box properties for the slice
        pca_bb_area = max(dim_x * dim_y, min_dim * min_dim * np.pi * 0.25)
        pca_bb_diag = max(np.sqrt(dim_x**2 + dim_y**2), min_dim)

    return dim_x, dim_y, pca_bb_diag, pca_bb_area

#Function for the histogram slices
def hist_dim_3D_slices(point_cloud_slices, normalize):
    # Initialize lists to store results for each slice
    slice_dimX = []
    slice_dimY = []
    slice_diag = []
    slice_area = []

    # Compute dimensions for each 3D point cloud slice
    for point_cloud_slice in point_cloud_slices:
        dim_x, dim_y, diag, area = compute_slice_dimensions_3D(point_cloud_slice)
        slice_dimX.append(dim_x)
        slice_dimY.append(dim_y)
        slice_diag.append(diag)
        slice_area.append(area)

    if(normalize):
        slice_dimX = [i / max(slice_dimX) for i in slice_dimX]
        slice_dimY = [i / max(slice_dimY) for i in slice_dimY]
        slice_diag = [i / max(slice_diag) for i in slice_diag]
        slice_area = [i / max(slice_area) for i in slice_area]

    return slice_dimX, slice_dimY, slice_diag, slice_area


def append_hist_to_data(data, hist, n_slices, key):
    for i in range(n_slices):
        key_nu = key + f'_{i+1}'   
        if key_nu not in data:
            data[key_nu] = []  # Initialize the key if not present
        if i < len(hist):
            # Add the count values for the current slice to the column
            data[key_nu].append(hist[i])
        else:
            # Add a placeholder or default value for missing data
            data[key_nu].append(None)


#%% File creation 

# Create a PyQt application instance
app = QApplication([])

# Ask the user to select a folder
folder_path = QFileDialog.getExistingDirectory(None, "Select Folder Containing Files", default_path, QFileDialog.ShowDirsOnly)

# Get a list of files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith('.laz')]

# Extract and sort file paths based on the numbers in brackets
sorted_files = sorted([os.path.join(folder_path, file) for file in files],
                      key=lambda x: int(x.split('(')[1].split(')')[0]) if '(' in x and ')' in x else float('inf'))

# Check if there are any valid files
if not sorted_files:
    print("No valid files found.")
else:
    # Read the Excel file with Ext_Class labels
    labels_df = pd.read_excel('D:\ODOT_SPR866\My Label Data Work\Sample Label data for testing\Ext_Class_labels.xlsx')  #Path to Ext_Class labels
    
    # Create a DataFrame to store the file paths, bounding box dimensions, and Ext_Class values
    data = {
            'File Paths': sorted_files, 
            'Total Points': [],
            'Ext_Class_Label': [],
            'Ext_class %': [], 
            'Ext_class': [], 
            'Root_class': [],
            'Sub_class': [],
            'In_Class_Prio': [],
            'Bound_Box Dim_X': [], 
            'Bound_Box Dim_Y': [], 
            'Bound_Box Dim_Z': [],
            'bb_centerX': [],
            'bb_centerY': [],
            'bb_centerZ': [],
            'pca_2d_dimX': [],
            'pca_2d_dimY': [],
            'pca_2d_dimZ': [],
            **{f'hist_npts_{i+1}': [] for i in range(n_slices)}, # Add separate columns for each element of hist_npts
            **{f'slice_dimX_{i+1}': [] for i in range(n_slices)}, # Add separate columns for each element of slice_dimX
            **{f'slice_dimY_{i+1}': [] for i in range(n_slices)}, # Add separate columns for each element of slice_dimY
            **{f'slice_diag_{i+1}': [] for i in range(n_slices)}, # Add separate columns for each element of slice_diag
            **{f'slice_area_{i+1}': [] for i in range(n_slices)}, # Add separate columns for each element of slice_area
            }

    # Count the total number of files
    total_files = len(sorted_files)
    
    # Initialize a counter for processed files
    processed_files = 0
    
    for file_path in sorted_files:
        # Read point cloud from LAS file
        point_cloud = readPtcloud(file_path)
        
        # Filter the XYZ ptc only
        xyzptc = xyz_ptc(point_cloud)
        
        #####
        min_xyz, max_xyz = compute_bounding_box_new(xyzptc) # Compute min & max XYZ
        bb_center_x, bb_center_y, bb_center_z = boundingbox_center(min_xyz, max_xyz)   # Compute the bounding box center  
        pca_2d_dimx, pca_2d_dimy, pca_2d_dimz = compute_pca_bounding_box(xyzptc)    # PCA bounding box
        hist_npts = hist_npts_xy(point_cloud_3D_slice(xyzptc, n_slices))   # Histogram no of points in each slice
        slice_dimX, slice_dimY, slice_diag, slice_area = hist_dim_3D_slices(point_cloud_3D_slice(xyzptc, n_slices), False)     # Slice histograms
        slice_dimX_norm, slice_dimY_norm, slice_diag_norm, slice_area_norm = hist_dim_3D_slices(point_cloud_3D_slice(xyzptc, n_slices), True)     # Slice histograms
        pca_slice_dimX, pca_slice_dimY, pca_slice_diag, pca_slice_area = hist_dim_3D_slices(point_cloud_3D_slice_pca(xyzptc, n_slices), False)     # Slice histograms
        pca_slice_dimX_norm, pca_slice_dimY_norm, pca_slice_diag_norm, pca_slice_area_norm = hist_dim_3D_slices(point_cloud_3D_slice_pca(xyzptc, n_slices), True)     # Slice histograms
    
        # Compute bounding box dimensions
        dimX, dimY, dimZ = compute_bounding_box(point_cloud)
        # Calculate and extract the Ext_Class value with its percentage
        ext_class_value, ext_class_percentage, root_class = calculate_majority_ext_class(point_cloud)
        

        #Ext Class Summary 2
        sum_ext_values, count, total_points = extract_summarized_values(point_cloud)
        
        # print("Summarized Values:", sum_ext_values)
        # print("Total Number of Points:", total_points)
        # print("Count of Differently Summarized Variables:", count)
      
        
        # Find matching Root_Ext_Class entries for each extracted Ext_Class value
        matching_entries = []
        #root_classes = [] 
        for ext_class_value in sum_ext_values:
            root_class1 = np.floor(ext_class_value/10000)*10000
            root_class2 = np.floor(ext_class_value/10000)
            
            ext_class_prefix = int(str(root_class1)[:3])
            # Extract the last three digits as Sub_Class
            ext_class_value = int(ext_class_value)
            sub_class = ext_class_value % 1000
                        
            # Find matching Root_Ext_Class entries
            matching_entries_for_value = labels_df[labels_df['Root_Ext_Class'].astype(str).str.startswith(str(ext_class_prefix))]
        
            # Check for duplicates and append unique values
            unique_entries = matching_entries_for_value['In_Class_Priority'].unique()
            if len(unique_entries) > 0:
                matching_entries.append(min(unique_entries))
            else:
                matching_entries.append(0)
        #Find the unique entires
        filtered_matching_entries = list(set(matching_entries))
        
        # Add data to the dictionary
        data['Total Points'].append(total_points)
        data['Ext_class'].append(ext_class_value)
        data['Ext_class %'].append(ext_class_percentage)
        data['Root_class'].append(root_class2)
        data['Sub_class'].append(sub_class)
        data['In_Class_Prio'].append(min(filtered_matching_entries)) #return the minimum value
        data['Bound_Box Dim_X'].append(dimX[0])
        data['Bound_Box Dim_Y'].append(dimY[0])
        data['Bound_Box Dim_Z'].append(dimZ[0])
        data['bb_centerX'].append(bb_center_x)
        data['bb_centerY'].append(bb_center_y)
        data['bb_centerZ'].append(bb_center_z)
        data['pca_2d_dimX'].append(pca_2d_dimx)
        data['pca_2d_dimY'].append(pca_2d_dimy)
        data['pca_2d_dimZ'].append(pca_2d_dimz)
        
        append_hist_to_data(data, hist_npts, n_slices, 'hist_npts')
        
        append_hist_to_data(data, slice_dimX, n_slices, 'slice_dimX') 
        append_hist_to_data(data, slice_dimY, n_slices, 'slice_dimY')
        append_hist_to_data(data, slice_diag, n_slices, 'slice_diag')
        append_hist_to_data(data, slice_area, n_slices, 'slice_area')
        
        append_hist_to_data(data, slice_dimX_norm, n_slices, 'slice_dimX_norm') 
        append_hist_to_data(data, slice_dimY_norm, n_slices, 'slice_dimY_norm')
        append_hist_to_data(data, slice_diag_norm, n_slices, 'slice_diag_norm')
        append_hist_to_data(data, slice_area_norm, n_slices, 'slice_area_norm')

        append_hist_to_data(data, pca_slice_dimX, n_slices, 'pca_slice_dimX') 
        append_hist_to_data(data, pca_slice_dimY, n_slices, 'pca_slice_dimY')
        append_hist_to_data(data, pca_slice_diag, n_slices, 'pca_slice_diag')
        append_hist_to_data(data, pca_slice_area, n_slices, 'pca_slice_area')

        append_hist_to_data(data, pca_slice_dimX_norm, n_slices, 'pca_slice_dimX_norm') 
        append_hist_to_data(data, pca_slice_dimY_norm, n_slices, 'pca_slice_dimY_norm')
        append_hist_to_data(data, pca_slice_diag_norm, n_slices, 'pca_slice_diag_norm')
        append_hist_to_data(data, pca_slice_area_norm, n_slices, 'pca_slice_area_norm')
        
        # Increment the counter for processed files
        processed_files += 1
    
        # Calculate the progress percentage
        progress_percentage = (processed_files / total_files) * 100
    
        # Print the progress
        print(f"Processing file {processed_files}/{total_files} - Progress: {progress_percentage:.2f}%")
        
        # # Add each element in hist_npts to separate columns
        # for i in range(n_slices):
        #     key = f'hist_npts_{i+1}'
        #     if key not in data:
        #         data[key] = []  # Initialize the key if not present
        #     if i < len(hist_npts):
        #         # Add the count values for the current slice to the column
        #         data[key].append(hist_npts[i])
        #     else:
        #         # Add a placeholder or default value for missing data
        #         data[key].append(None)
        
        # # Add each element in slice_dimX to separate columns
        # for i in range(n_slices):
        #     key = f'slice_dimX_{i+1}'
        #     if key not in data:
        #         data[key] = []  # Initialize the key if not present
        #     if i < len(slice_dimX):
        #         # Add the count values for the current slice to the column
        #         data[key].append(slice_dimX[i])
        #     else:
        #         # Add a placeholder or default value for missing data
        #         data[key].append(None)
        
        # # Add each element in slice_dimY to separate columns
        # for i in range(n_slices):
        #     key = f'slice_dimY_{i+1}'
        #     if key not in data:
        #         data[key] = []  # Initialize the key if not present
        #     if i < len(slice_dimY):
        #         # Add the count values for the current slice to the column
        #         data[key].append(slice_dimY[i])
        #     else:
        #         # Add a placeholder or default value for missing data
        #         data[key].append(None)  
        
        # # Add each element in slice_dimY to separate columns
        # for i in range(n_slices):
        #     key = f'slice_diag_{i+1}'
        #     if key not in data:
        #         data[key] = []  # Initialize the key if not present
        #     if i < len(slice_diag):
        #         # Add the count values for the current slice to the column
        #         data[key].append(slice_diag[i])
        #     else:
        #         # Add a placeholder or default value for missing data
        #         data[key].append(None) 
        
        # # Add each element in slice_dimY to separate columns
        # for i in range(n_slices):
        #     key = f'slice_area_{i+1}'
        #     if key not in data:
        #         data[key] = []  # Initialize the key if not present
        #     if i < len(slice_area):
        #         # Add the count values for the current slice to the column
        #         data[key].append(slice_area[i])
        #     else:
        #         # Add a placeholder or default value for missing data
        #         data[key].append(None)
            
    # Add a new column 'Ext_Class_Label' to the DataFrame with labels
    data['Ext_Class_Label'] = add_labels(data['Ext_class'], labels_df)
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Extract the folder name from the folder_path
    folder_name = os.path.basename(folder_path)

    # Get the parent directory (one step outside the selected folder)
    parent_directory = os.path.dirname(folder_path)

    # Generate the Excel file name with the folder name
    csv_file_name = f"{folder_name}_data.csv"

    # Combine the parent directory and file name
    csv_file_path = os.path.join(parent_directory, csv_file_name)

    # Save the DataFrame to an Excel file
    df.to_csv(csv_file_path, index=False)

    print(f"File paths, bounding box dimensions, and Ext_Class values saved to {csv_file_path}")
