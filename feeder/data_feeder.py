import os
import numpy as np

# open and get the file list name (.npz files)
def get_npz_data_file_name(path_to_file: str):
    with open(file=path_to_file) as f:
        file_name_list = f.read().splitlines(keepends=False)
        
        if len(file_name_list) != 0:
            return file_name_list
        else:
            raise Exception("data file is empty")

# feed the raw data to the data class => model
def feed_data(raw_data_path: str, file_list_path: str):
    """Load raw data (images and poses) from .npz files define in file_list_path 
    
    Args:
        raw_data_path (str): path to raw data files which is .npz files
        file_list_path (str): path to file list files which contains name of .npz files
    """
    
    npz_file_name_list = get_npz_data_file_name(path_to_file=file_list_path)
    
    poses = []
    images = []
    
    for i in range (len(npz_file_name_list)):
        file_path = os.path.join(raw_data_path, npz_file_name_list[i])
        data = np.load(file=file_path, allow_pickle=True)
        
        data_image = np.array(data["image"]) # get the numerical image data (d0, d1, d2, d3) - (number of images, pixel, pixel, channel) 
        
        data_pose = np.array(data["pose"]) # get the pose data (d0, d1) - (number of images, euler angles[yaw, pitch, roll])
        
        standardize_pose = np.clip(data_pose, -99, 99) # standardize to make all the pose value to be around -99 to 99 
        
        poses.append(standardize_pose)
        images.append(data_image)
                 
    poses = np.concatenate(poses) # when add to original poses array -> only one index -> remove the [ [...] ] outside square brackets -> total pose index
    images = np.concatenate(images) # when add to original images array -> only one index -> remove the [ [...] ] outside square brackets -> total image index
    
    return images, poses
    
    