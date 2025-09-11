from torch.utils.data.dataset import Dataset
from torchvision import transforms
from feeder.data_feeder import feed_data
from feeder.data_augmentation import Cutout
from PIL import Image

import numpy as np
import torch


class Pose_300W_LP(Dataset):
    def __init__(self, raw_data_path, file_list_path, transform=None):
        self.raw_data_path = raw_data_path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.2), antialias=True),
                transforms.Normalize(0, 1)
            ]
        ) if transform is None else transform
        self.file_list_path = file_list_path
        self.images, self.poses = feed_data(self.raw_data_path, self.file_list_path)
        self.length = len(self.poses)
        self.cut_out = Cutout()
        
    def __getitem__(self, index):
        pose = self.poses[index]
        img = self.transform(Image.fromarray(self.images[index].astype(np.uint8)))
        img = self.cut_out(img=img)
        
        yaw = -torch.tensor([pose[0]]) # negative for standardize between dataset and model to learn
        pitch = torch.tensor([pose[1]])
        roll = torch.tensor([pose[2]])
        
        euler_label = torch.FloatTensor([yaw.item(), pitch.item(), roll.item()])
        
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw.item(), pitch.item(), roll.item()], bins) - 1
        binned_pose = np.clip(binned_pose, 0, 65)
        labels = torch.LongTensor(binned_pose)   
        
        return img, euler_label, labels, index

    def __len__(self):
        return self.length
    
class AFLW2000(Dataset):
    def __init__(self, raw_data_path, file_list_path, transform=None):
        self.raw_data_path = raw_data_path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ]
        ) if transform is None else transform
        self.file_list_path = file_list_path
        self.images, self.poses = feed_data(self.raw_data_path, self.file_list_path)
        self.length = len(self.poses)
        
    def __getitem__(self, index):
        pose = self.poses[index]
        img = self.transform(Image.fromarray(self.images[index].astype(np.uint8)))
        
        yaw = -torch.tensor([pose[0]]) # negative for standardize between dataset and model to learn
        pitch = torch.tensor([pose[1]])
        roll = torch.tensor([pose[2]])
        
        euler_label = torch.FloatTensor([yaw.item(), pitch.item(), roll.item()])
        
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw.item(), pitch.item(), roll.item()], bins) - 1
        binned_pose = np.clip(binned_pose, 0, 65)
        labels = torch.LongTensor(binned_pose)   
        
        return img, euler_label, labels, index

    def __len__(self):
        return self.length

class BIWI(Dataset):
    def __init__(self, raw_data_path, file_list_path, transform=None):
        self.raw_data_path = raw_data_path
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0, 1)
            ]
        ) if transform is None else transform
        self.file_list_path = file_list_path
        self.images, self.poses = feed_data(self.raw_data_path, self.file_list_path)
        self.length = len(self.poses)
        
    def __getitem__(self, index):
        pose = self.poses[index]
        img = self.transform(Image.fromarray(self.images[index].astype(np.uint8)))
        
        yaw = -torch.tensor([pose[0]]) # negative for standardize between dataset and model to learn
        pitch = torch.tensor([pose[1]])
        roll = torch.tensor([pose[2]])
        
        euler_label = torch.FloatTensor([yaw.item(), pitch.item(), roll.item()])
        
        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw.item(), pitch.item(), roll.item()], bins) - 1
        binned_pose = np.clip(binned_pose, 0, 65)
        labels = torch.LongTensor(binned_pose)   
        
        return img, euler_label, labels, index

    def __len__(self):
        return self.length