import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
from utils.config import base_dir, subfolders

class TimeVariantDataAugmentation:
    @staticmethod
    def window_slicing(data, slice_percentage=0.9):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        orig_width = data.shape[1]
        slice_width = int(orig_width * slice_percentage)
        max_start = orig_width - slice_width
        start_point = torch.randint(0, max_start + 1, (1,)).item()
        sliced_data = data[:, start_point:start_point + slice_width]

        augmented_data = torch.nn.functional.interpolate(
            sliced_data.unsqueeze(0), 
            size=orig_width, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        return augmented_data

    @staticmethod
    def window_warping(data, warping_factors=[0.5, 2.0]):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        orig_width = data.shape[1]
        window_width = int(orig_width * 0.1)
        start_point = torch.randint(0, orig_width - window_width + 1, (1,)).item()
        warping_factor = np.random.choice(warping_factors)
        window = data[:, start_point:start_point + window_width]

        warped_window_width = int(window_width * warping_factor)
        warped_window = torch.nn.functional.interpolate(
            window.unsqueeze(0), 
            size=warped_window_width, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)

        augmented_data = data.clone()
        end_point = start_point + warped_window_width
        if warped_window_width < window_width:
            augmented_data[:, start_point:end_point] = warped_window
        else:
            augmented_data[:, start_point:start_point + window_width] = warped_window[:window_width]
        
        return augmented_data

    @staticmethod
    def jittering(data, mean=0, std_dev=0.03):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        noise = torch.normal(mean, std_dev, size=data.shape)
        return data + noise

class StochasticAugmentation:
    def __init__(self, augmentation_methods):
        self.augmentation_methods = augmentation_methods
    
    def __call__(self, data):
        method = np.random.choice(self.augmentation_methods)
        return method(data)

class CustomImageDataset(Dataset):
    def __init__(self, base_dir, subfolders, transform=None, augmentations=None):
        self.base_dir = base_dir
        self.subfolders = subfolders
        self.transform = transform
        self.augmentations = augmentations  # List of augmentation methods
        self.image_paths = []
        self.labels = []

        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(subfolders)
        
        # Collect image paths and labels
        for subfolder in subfolders:
            folder_path = os.path.join(base_dir, subfolder)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(subfolder)
        
        # Encode labels
        self.labels = self.label_encoder.transform(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply standard transformations like resizing, flipping, etc.
        if self.transform:
            image = self.transform(image)
        
        # Apply time-variant augmentations if any are specified
        if self.augmentations:
            for aug in self.augmentations:
                image = aug(image)

        return image, label

augmentation_methods = [
    TimeVariantDataAugmentation.window_slicing,
    TimeVariantDataAugmentation.window_warping,
    TimeVariantDataAugmentation.jittering
]

# Define the stochastic augmentation function
stochastic_augmentation = StochasticAugmentation(augmentation_methods)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset(base_dir, subfolders, transform=transform, augmentations=[stochastic_augmentation])

train_size = int(0.85 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Manually shuffle the training dataset
torch.manual_seed(42)
train_indices = torch.randperm(len(train_dataset))
train_dataset = Subset(train_dataset, train_indices)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
