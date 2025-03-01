import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from data.dataset import CustomImageDataset, transform, stochastic_augmentation
from model.model_vgg import initialize_vgg16, initialize_vgg19
from model.model_customcnn import CustomCNN
from utils.train import train_model, test_model
from utils.config import device, num_epochs, criterion, scheduler_vgg19, optimizer_vgg19
from data.dataset import train_loader

# Initialize model
model = initialize_vgg19(num_classes=6).to(device)
model_name = f"best_model_CustomVGG.pt"

# Check if pre-trained model exists
if os.path.exists(model_name):
    print(f"Loading pre-trained model: {model_name}")
    model.load_state_dict(torch.load(model_name))
else:
    print(f"No pre-trained model found. Training a new model...")
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_vgg19
    scheduler = scheduler_vgg19

    # Train the model
    train_model(model, train_loader, train_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs)

# Test the model
test_model(model, train_loader, criterion, device)
