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

def create_image_labels_list(data_path) -> list:
    """
    Given, the data path, return a list image_labels, containing tuples (image_file_name (str), category_labels (list of 0,1s))

    Args:
        data_path (str)

    Returns:
        image_labels (list): List of 0,1s corresponding to class labels.     
    """
    
    # create image labels from annotations
    with open(data_path + "category.json", "r") as f:
        categories = json.load(f)
    with open(data_path + "metadata.json", "r") as f:
        metadata = json.load(f)
        
    category_indices = {cat['name']: idx for idx, cat in enumerate(categories)}

    image_labels = {}

    for image, values in metadata.items():
        image_file_name = values["filename"]
        
        if image_file_name not in image_labels:
            image_labels[image_file_name] = [0] * len(categories)
            
        for cat in values["categories"]:
            cat_index = category_indices[cat]
            image_labels[image_file_name][cat_index] = 1
            
    image_labels = list(image_labels.items())
    return image_labels


# split data
def split_data(
    image_labels,
    train_size,
    val_size,
    test_size,
    random_state=0
):
    # split test and val data
    train_val_data, test_data = train_test_split(image_labels, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(train_val_data, test_size=val_size, random_state=random_state)
    
    # based on how much train data we want to use
    _, train_data = train_test_split(train_data, test_size=train_size, random_state=random_state+train_size)
    
    return train_data, val_data, test_data


# our Torch Dataset object
class MultiLabelDataset(Dataset):
    def __init__(self, image_labels, root_dir, transform=None):
        """
        Args:
            image_labels (list of tuples): List of tuples (image_path, label_vector).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_labels = image_labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_labels[idx][0])
        image = Image.open(img_name)
        labels = self.image_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(labels)
    
    
# create train, val, test Torch DataLoaders
def create_data_loaders(
    train_data,
    val_data,
    test_data,
    image_root_dir,
    batch_size,
    transform
):
    train_dataset = MultiLabelDataset(image_labels=train_data, root_dir=image_root_dir, transform=transform)
    val_dataset = MultiLabelDataset(image_labels=val_data, root_dir=image_root_dir, transform=transform)
    test_dataset = MultiLabelDataset(image_labels=test_data, root_dir=image_root_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


