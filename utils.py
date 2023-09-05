import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from PIL import Image
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset, random_split
import csv
from torchvision.models import resnet18
import json
from torch.utils.data import Subset

class Predictor(nn.Module):

    """
    A class which create a neural network model
    """

    def __init__(self):
        """
        Model creation

        """

        super().__init__()

        self.resnet18 = resnet18(weights=None, progress=False).eval()
        num_ftrs = (
            self.resnet18.fc.in_features
        )  # Get the number of input features for the last layer

        self.resnet18.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2),
            # nn.ReLU(),
            # nn.Linear(512, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce an output y with a tensor input x

        Returns :
        -------------
        y_pred : tensor
            model's prediction

        """

        y_pred = self.resnet18(x)
        return y_pred

    def reset_weights(self):
        """
        Reset the model's weight
        """

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        # Apply the weight reset to all Conv2d and Linear layers
        self.apply(weight_reset)


class LinearClassifier(torch.nn.Module):
    """
    A class which create a linear model
    """

    def __init__(self, input_dim=150528, output_dim=2):
        """
        Model creation


        Parameters:
        --------------------------------
        input_dim : int
            dimension of the tensor's input
        output_dim : int
            dimension of the tensor's output
        """

        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Produce an output y with a tensor input x

        Returns :
        -------------
        y_pred : tensor
            model's prediction
        """
        x = self.linear(x)
        return x

    def reset_weights(self):
        """
        Reset the model's weight
        """

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        # Apply the weight reset to all Conv2d and Linear layers
        self.apply(weight_reset)


class ConvertToRGB(object):
    """
    Class which convert images in 3 canal (Red, Green, Blue)
    """

    def __call__(self, img):
        """
        Convert images in RGB

        Returns :
        -------------
        img : tensor
            images with 3 canals
        """
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


class CustomDataset(torch.utils.data.Dataset):

    """
    Class which create a custom dataset
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Dataset creation

        Parameters:
        --------------------------------
        images_paths : str
            paths for each image
        labels : int
            label for each images,
        transform : torchvision.transforms.Compose
            transformation applied to the tensors
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = (
            F.one_hot(torch.tensor(label), num_classes=2).squeeze().to(torch.float32)
        )
        return image, label

class Strategie():

    def __init__(self, budget, n):
        # initialiser les quantités
        self.budget = budget
        self.n = n
        self.counter = [0]*n #nombre de classe

    def reset(self):
        # reset les compteurs, etc
        self.counter = [0]*self.n

    def get_action(self, yt):
            # procédure de sélection d'une action
            if self.counter[yt.item()]< self.budget/2:
                return 1
            else:
                return 0
    
    def update(self, yt):
        #mettre a jour les compteurs 
        self.counter[yt.item()] += 1
        print(self.counter)

class SelectedImagesDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, labels = self.data[idx]
        return inputs, labels
    
def iterate_data(
    dataloader,
    model,
    criterion,
    device,
    optimizer=None,
    is_training=False,
    scheduler=None,
):
    """
    A function that allows to iterate the data and train or not the model

    Parameters:
    --------------------------------
    model : LinearClassifier or Predictor
        model choosen for the train
    criterion : torch.nn.Module
        loss function
    device : torch.device
        The device that runs the model
    optimizer : torch.optim
        optimizer for the model
    is_training : boolean
        If True, train the model, else just do a validation process
    scheduler

    Return:
    ---------------------------------
    mean_loss : tensor
        mean loss for one iteration
    mean_accuracy : tensor
        mean accuracy for one iteration
    """
    loss_list = []
    total_samples = 0
    correct_predictions = 0

    if is_training:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            output = model(images)
            loss = criterion(output, labels)
            loss_list.append(loss.item())  # Ajouter la perte actuelle à la liste

            total_samples += images.size(0)  # Accumuler le nombre total d'échantillons

            labels_ = torch.argmax(labels, dim=1)
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (
                (predicted == labels_).sum().item()
            )  # Accumuler le nombre de prédictions correctes

            if is_training:
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()

    mean_loss = torch.tensor(loss_list).mean().item()
    mean_accuracy = (100.0 * correct_predictions) / total_samples
    return mean_loss, mean_accuracy

def save_matrix_to_csv(filename, matrix, num_repetitions, args):
    """
    A function which save results to csv

    Parameters:
    --------------------------------
    filename : str
        name of the path where we want to save the dataframe
    matrix : list
        matrix with all results we want to save
    num_repetitions : int
        number of repetition we used
    """

    # Assurez-vous que chaque élément est une sous-liste
    matrix = [[x] if not isinstance(x, list) else x for x in matrix]

    # Trouver la longueur maximale de sous-liste
    max_size = max(len(sublist) for sublist in matrix)

    # Compléter chaque sous-liste avec des nan pour atteindre cette longueur
    matrix = [sublist + [np.nan] * (max_size - len(sublist)) for sublist in matrix]

    matrix = np.array(matrix).T

    with open(os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), filename), "w", newline="") as file:
        writer = csv.writer(file)

        header = ["Epoch"] + [f"Repetition {i+1}" for i in range(num_repetitions)]
        writer.writerow(header)

        for epoch, repetitions in enumerate(matrix):
            row_data = [epoch] + list(repetitions)
            writer.writerow(row_data)


def compute_std_for_matching_rows(matrix1, matrix2, matrix3, matrix4, axis=0):
    # Trouvez la taille de la sous-liste la plus longue parmi toutes les matrices
    max_size = max(max(len(sublist) for sublist in matrix) for matrix in [matrix1, matrix2, matrix3, matrix4])
    
    # Remplissez chaque sous-liste avec des 'nan' pour qu'elles aient toutes la même taille
    def pad_with_nan(sublist):
        return sublist + [np.nan] * (max_size - len(sublist))
    
    matrix1 = [pad_with_nan(sublist) for sublist in matrix1]
    matrix2 = [pad_with_nan(sublist) for sublist in matrix2]
    matrix3 = [pad_with_nan(sublist) for sublist in matrix3]
    matrix4 = [pad_with_nan(sublist) for sublist in matrix4]

    # Convertissez chaque liste remplie en tableau numpy
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)
    matrix3 = np.array(matrix3)
    matrix4 = np.array(matrix4)

    # Calculez et renvoyez l'écart type pour chaque matrice
    std1 = np.std(matrix1, axis=axis)
    std2 = np.std(matrix2, axis=axis)
    std3 = np.std(matrix3, axis=axis)
    std4 = np.std(matrix4, axis=axis)
    
    return std1, std2, std3, std4

def save_results_to_csv(filename, online_loss, online_accuracy, test_loss, test_accuracy, args):
    """
    A function which saves results to csv.

    Parameters:
    --------------------------------
    filename : str
        name of the path where we want to save the dataframe
    online_loss : list
        list with all value for the training loss for the train set
    online_accuracy : list
        list with all value for the training accuracy for the train set
    test_loss: list
        list with all value for the validation loss for the validation set
    test_accuracy : list
        list with all value for the validation accuracy for the validation set
    """

    # Assurez-vous que chaque élément est une sous-liste
    online_loss = [[x] if not isinstance(x, list) else x for x in online_loss]
    online_accuracy = [[x] if not isinstance(x, list) else x for x in online_accuracy]
    test_loss = [[x] if not isinstance(x, list) else x for x in test_loss]
    test_accuracy = [[x] if not isinstance(x, list) else x for x in test_accuracy]

    # Trouver la longueur maximale de sous-liste parmi toutes les matrices
    max_size = max(max(len(sublist) for sublist in matrix) for matrix in [online_loss, online_accuracy, test_loss, test_accuracy])

    # Compléter chaque sous-liste avec des nan pour atteindre cette longueur
    online_loss = [sublist + [np.nan] * (max_size - len(sublist)) for sublist in online_loss]
    online_accuracy = [sublist + [np.nan] * (max_size - len(sublist)) for sublist in online_accuracy]
    test_loss = [sublist + [np.nan] * (max_size - len(sublist)) for sublist in test_loss]
    test_accuracy = [sublist + [np.nan] * (max_size - len(sublist)) for sublist in test_accuracy]

    # Convertir chaque liste complétée en tableau numpy
    online_loss = np.array(online_loss)
    online_accuracy = np.array(online_accuracy)
    test_loss = np.array(test_loss)
    test_accuracy = np.array(test_accuracy)
    data = zip(online_loss, online_accuracy, test_loss, test_accuracy)

    with open(os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), filename), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])
        for i, row in enumerate(data):
            writer.writerow([i] + list(row))


    with open(os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), filename), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"]
        )
        for i, row in enumerate(data):
            writer.writerow([i] + list(row))