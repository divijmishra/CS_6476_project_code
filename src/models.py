import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.optim import Adam
from tqdm import tqdm


def get_model(
    backbone,
    tune_conv,
    num_categories,
    dropout_rate=0.5
):
    """
    Returns a PyTorch model based on our experiment configuration.
    If tune_conv is False, only the fully-connected layers are trained. Else, the last conv-blocks (refer to the code) are trained.
    
    Args:
        backbone: "vgg16" or "resnet50",
        tune_conv: Boolean, 
        num_categories: Number of category labels in our problem.
        
    Returns:
        model: PyTorch model. 
    """
    
    # VGG-16
    if backbone == "vgg16":
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad=False
            
        num_features = model.classifier[0].in_features
        
        # added one more linear layer of 512 nodes before the final layer
        model.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_categories),
            nn.Sigmoid()
        )
        
        for param in model.classifier.parameters():
            param.requires_grad=True
        
        # set last 3 conv layers to train, if needed
        if tune_conv:
            conv_layer_indices_to_train = [24, 26, 28]
            for idx, module in enumerate(model.features.children()):
                if idx in conv_layer_indices_to_train:
                    for param in module.parameters():
                        param.requires_grad = True
                        
        return model
        
    # ResNet-50
    elif backbone == "resnet50":
        model = models.resnet50(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad=False
            
        num_features = model.fc.in_features
        
        # added one more linear layer of 512 nodes before the final layer
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_categories),
            nn.Sigmoid()
        )
        
        for param in model.fc.parameters():
            param.requires_grad=True
            
        # set layer4 conv layers to train, if needed
        if tune_conv:
            for param in model.layer4.parameters():
                param.requires_grad=True
                
        return model
        
    else:
        raise ValueError("Model not supported. Pick \"vgg16\" or \"resnet50\". ")