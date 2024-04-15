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
import time

from utils import (
    create_image_labels_list,
    split_data,    
    create_data_loaders,
    plot_training_history,
    show_images_with_predictions,
    save_metrics
)
from models import get_model

print("Imports completed.")
code_start_time = time.time()


############ CONFIG OPTIONS
##############################################################

# How much train data do we want to use?
##
train_size = 512  # 10000 or 40000
##

# define model
##
backbone = "resnet50"  # "resnet50"/"vgg16" supported, not using "vgg16"
##

##
tune_conv = False  # True or False
##

# # options for hyper-parameter tuning
# num_epochs_optuna = 4
# num_trials_optuna = 6

# options for final training
##
lr = 1e-4# (1e-3, 0.5e-4,  1e-4)
##
num_epochs = 30
batch_size = 32

patience = 5  # number of epochs to wait before early stopping

##############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

seed = 42

# data path
# data_path = "~/scratch/CS_6476_project_code/data/processed_data/"
data_path = "data/processed_data/"

# path to images
image_root_dir = data_path + "images/"

# run saving path
run_path = f"runs/{backbone}_tuneconv={tune_conv}_data={train_size}_lr={lr}_num_epochs={num_epochs}_run01/"
while os.path.exists(run_path):
    run_index = int(run_path[-3:-1]) + 1
    run_path = run_path[:-3] + f"{run_index:02d}/"
os.makedirs(run_path)
print(f"This run will be saved in {run_path}.")

model_path = "models" + run_path[4:]
os.makedirs(model_path)

print("Config settings completed.")


############ DATA PRE-PROCESSING

# create image labels from annotations
image_labels, categories = create_image_labels_list(data_path)
# num_categories = len(image_labels[0][1])
num_categories = 1  # vulnerable or not

# split data
# train_size defined earlier
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

train_loader, val_loader, test_loader = create_data_loaders(
    train_data, val_data, test_data, image_root_dir, batch_size, transform
)

print("Data preprocessing completed.")


############ HYPERPARAMETER TUNING

loss_function = nn.BCELoss()

'''
def objective(trial):

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    dropout_rate = 0.5

    model = get_model(backbone, tune_conv, num_categories, dropout_rate)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs_optuna):
        epoch_start_time = time.time()
        
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

        epoch_end_time = time.time()
        
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_running_loss/len(val_loader)}, Time taken: {(epoch_end_time - epoch_start_time)/60.0} minutes.")

    print('Finished Training')

    return val_running_loss

study = optuna.create_study(direction='minimize')
print("Beginning hyperparameter tuning.")

hyp_start_time = time.time()
study.optimize(objective, n_trials=num_trials_optuna)
hyp_end_time = time.time()

print("Hyperparameter tuning completed.")
print(f"Time taken: {(hyp_end_time - hyp_start_time)/60.0} minutes.")
print("Best hyperparameters: ", study.best_trial.params)
'''


############ TRAIN THE MODEL

loss_function = nn.BCELoss()

# best hyperparameters
# lr = study.best_trial.params['lr']
# dropout_rate = study.best_trial.params['dropout_rate']

# lr defined in config at the top
dropout_rate = 0.5

# write hyperparameters to a txt file
content = f"""backbone: {backbone}
tune_conv: {tune_conv}
train_size: {train_size}
batch_size: {batch_size}
num_epochs: {num_epochs}
lr: {lr}
dropout_rate: {dropout_rate}"""

''' Insert these above if you re-enable Optuna
num_epochs_optuna: {num_epochs_optuna}
num_trials_optuna: {num_trials_optuna}
'''

with open(run_path + "hyperparameters.txt", "w") as file:
    file.write(content)

# define the model
model = get_model(backbone, tune_conv, num_categories, dropout_rate)
model = model.to(device)

# define the optimizer
optimizer = Adam(model.parameters(), lr=lr)

# train the model
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': []
}

train_start_time = time.time()

# for early stopping
best_val_loss = float('inf')
counter = 0  # counter for tracking epochs without improvement
# patience defined above
# patience = 5  # number of epochs to wait before early stopping

for epoch in range(num_epochs):
    final_epochs = epoch
    epoch_start_time = time.time()
    
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

    epoch_end_time = time.time()

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
    
    # epoch_end_time = time.time()

    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}, Validation Accuracy: {epoch_val_accuracy}, Training time taken: {(epoch_end_time - epoch_start_time)/60.0} minutes.")

    # Recording the metrics for this epoch
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_accuracy)
    
    # Check for early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
train_end_time = time.time()

# save history
with open(run_path + "history.json", 'w') as file:
    json.dump(history, file)

print('Model training completed.')
print(f"Time taken: {(train_end_time - train_start_time)/60.0} minutes")


############ SAVE THE MODEL

torch.save(model.state_dict(), model_path + 'model.pth')
print("Saved the model.")


############ SAVING METRICS

# save accuracy/loss metrics
plot_training_history(history, run_path)

# save test images
show_images_with_predictions(test_loader, model, device, categories, num_images=6, save_dir=run_path)

# save weighted metrics
label_weights = np.array([
    1.0,  # animal
    0.1,  # flat.driveable_surface
    1.0,  # human.pedestrian.adult
    1.0,  # human.pedestrian.child
    1.0,  # human.pedestrian.construction_worker
    1.0,  # human.pedestrian.personal_mobility
    1.0,  # human.pedestrian.police_officer
    1.0,  # human.pedestrian.stroller
    1.0,  # human.pedestrian.wheelchair
    0.2,  # movable_object.barrier
    0.2,  # movable_object.debris
    0.3,  # movable_object.pushable_pullable
    0.2,  # movable_object.trafficcone
    0.1,  # static_object.bicycle_rack
    0.5,  # vehicle.bicycle
    0.4,  # vehicle.bus.bendy
    0.4,  # vehicle.bus.rigid
    0.4,  # vehicle.car
    0.4,  # vehicle.construction
    0.4,  # vehicle.ego
    0.6,  # vehicle.emergency.ambulance
    0.6,  # vehicle.emergency.police
    0.5,  # vehicle.motorcycle
    0.4,  # vehicle.trailer
    0.4   # vehicle.truck
])
save_metrics(model, device, test_loader, label_weights, run_path)


code_end_time = time.time()

# print(f"Time taken for hyperparameter tuning: {(hyp_end_time - hyp_start_time)/60.0} minutes.")
print(f"Final epochs trained: {final_epochs + 1}")
print(f"Time taken for training: {(train_end_time - train_start_time)/60.0} minutes.")
print(f"Total time taken: {(code_end_time - code_start_time)/60} minutes.")

with open(run_path + "times.txt", "w") as file:
    # file.write(f"Time taken for hyperparameter tuning: {(hyp_end_time - hyp_start_time)/60.0} minutes.\n")
    file.write(f"Time taken for training: {(train_end_time - train_start_time)/60.0} minutes.\n")
    file.write(f"Total time taken: {(code_end_time - code_start_time)/60} minutes.\n")