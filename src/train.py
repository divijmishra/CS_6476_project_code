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
import optuna
from tqdm import tqdm

from utils import (
    create_image_labels_list,
    split_data,    
    create_data_loaders,
)
from models import get_model

print("Imports completed.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# data_path = "~/scratch/CS_6476_project_code/data/processed_data/"
data_path = "data/processed_data/"
seed = 42


##############################################################
############ CONFIG OPTIONS

# How much train data do we want to use?
train_size = 10000  # 10000 or 40000

# define model
backbone = "vgg16"  # "vgg16" or "resnet50"
tune_conv = False  # True or False

# options for hyper-parameter tuning
num_epochs_optuna = 10
num_trials_optuna = 16

# options for final training
num_epochs = 30

##############################################################


############ DATA PRE-PROCESSING

# create image labels from annotations
image_labels = create_image_labels_list(data_path)
num_categories = len(image_labels[0][1])

# split data
##############################################################
# # How much train data do we want to use?
# train_size = 10000
# # train_size = 20000
##############################################################
test_size, val_size = 2000, 2000
train_data, val_data, test_data = split_data(
    image_labels, train_size, val_size, test_size, seed
)

# create Torch DataLoaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_root_dir = data_path + "images/"
batch_size=32
train_loader, val_loader, test_loader = create_data_loaders(
    train_data, val_data, test_data, image_root_dir, batch_size, transform
)

##############################################################
# # define model
# backbone = "vgg16"  # "vgg16" or "resnet50"
# tune_conv = False  # True or False
##############################################################
loss_function = nn.BCELoss()


############ HYPERPARAMETER TUNING

def objective(trial):

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)

    model = get_model(backbone, tune_conv, num_categories, dropout_rate)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs_optuna):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_function(outputs, labels)

                val_running_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_running_loss/len(val_loader)}")

    print('Finished Training')

    return val_running_loss

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=num_trials_optuna)

print("Best hyperparameters: ", study.best_trial.params)


############ TRAIN THE MODEL

# best hyperparameters
lr = study.best_trial.params['lr']
dropout_rate = study.best_trial.params['dropout_rate']

model = get_model(backbone, tune_conv, num_categories, dropout_rate)
model = model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

# train the model
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': []
}

for epoch in range(num_epochs):
    # Training phase
    model.train() 
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            
            val_running_loss += loss.item()

            # Calculate accuracy
            predicted = outputs > 0.5  # Using 0.5 as threshold
            correct_preds += (predicted == labels.byte()).sum().item()
            total_preds += labels.size(0) * labels.size(1)

    epoch_train_loss = running_loss / len(train_loader)
    epoch_val_loss = val_running_loss / len(val_loader)
    epoch_val_accuracy = correct_preds / total_preds

    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_accuracy}")

    # Recording the metrics for this epoch
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_accuracy)

print('Finished Training')


############ SAVING METRICS