import torch
from torch import nn
from torch import optim
from model.model_vgg import initialize_vgg16, initialize_vgg19
from model.model_customcnn import CustomCNN
# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
import os
base_dir = os.getenv('DIAT_DATASET_PATH', r'D:\micro-doppler based target classification\Micro-Doppler-Based-Target-Classification-\ml_model\notebooks\DIAT-uSAT_dataset')
subfolders = [
    r"3_long_blade_rotor", 
    r"3_short_blade_rotor", 
    r"Bird", 
    r"Bird+mini-helicopter", 
    r"drone", 
    r"rc_plane", 
]

# Hyperparameters
batch_size = 32
learning_rate = 0.0001
num_epochs = 15

# Model names
model_vgg16_name = "vgg16"
model_vgg19_name = "vgg19"
model_cnn_name = "cnn"

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_vgg16 = optim.Adam(initialize_vgg16(num_classes=6).classifier.parameters(), lr=0.0001)
optimizer_vgg19 = optim.Adam(initialize_vgg19(num_classes=6).classifier.parameters(), lr=0.0001)

# Learning rate scheduler
scheduler_vgg16 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vgg16, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)
scheduler_vgg19 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vgg19, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)

# Loss and optimizer for Deeper CNN
model_deepercnn = CustomCNN(num_classes=6)
model_deepercnn.to(device)
optimizer_deepercnn = optim.Adam(model_deepercnn.parameters(), lr=0.0001)
scheduler_deepercnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_deepercnn, mode='min', factor=0.5, patience=2, min_lr=1e-7, verbose=True)