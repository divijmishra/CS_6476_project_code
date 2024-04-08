import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    return image_labels, categories


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


# save accuracy and loss curve
def plot_training_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Training Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_dir + "loss_accuracy.jpeg")
    
    
def tensor_to_image(tensor):
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    tensor = tensor.to('cpu').detach().numpy().transpose((1, 2, 0))
    tensor = np.clip(tensor, 0, 1)
    return tensor


# save test images
def show_images_with_predictions(dataloader, model, device, categories, num_images=6, save_dir=None):
    model.eval()
    images_so_far = 0
    plt.figure(figsize=(30, 30))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')

                predicted_labels = (outputs[j] > 0.5).int()
                pred_labels_text = [categories[idx]["name"] for idx, label in enumerate(predicted_labels) if label == 1]
                true_labels_text = [categories[idx]["name"] for idx, label in enumerate(labels[j]) if label == 1]

                ax.set_title(f"True: {true_labels_text}\nPred: {pred_labels_text}")
                plt.imshow(tensor_to_image(inputs.cpu().data[j]))

                if images_so_far == num_images:
                    model.train()
                    return
        model.train()
        
    plt.savefig(save_dir + "image_tests.jpg")
    
    
# save metrics
def save_metrics(model, device, data_loader, label_weights, save_dir):
    model.eval()

    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = (outputs > 0.5).float()

            all_preds.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_true_labels = np.array(all_true_labels)

    sample_weights = np.dot(all_true_labels, label_weights)

    precision = precision_score(all_true_labels, all_preds, average='samples', sample_weight=sample_weights)
    recall = recall_score(all_true_labels, all_preds, average='samples', sample_weight=sample_weights)
    f1 = f1_score(all_true_labels, all_preds, average='samples', sample_weight=sample_weights)

    # If we dont want to weight
    # precision = precision_score(all_true_labels, all_preds, average='micro')
    # recall = recall_score(all_true_labels, all_preds, average='micro')
    # f1 = f1_score(all_true_labels, all_preds, average='micro')

    # print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    with open(save_dir + "metrics.txt", "w") as file:
        file.write(f"Precision: {precision}")
        file.write(f"Recall: {recall}")
        file.write(f"F1 Score: {f1}")