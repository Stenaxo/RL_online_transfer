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
from torch.utils.data import DataLoader
import csv
from torchvision.models import resnet18
from utils import Predictor, LinearClassifier, ConvertToRGB, CustomDataset, Strategie, SelectedImagesDataset, iterate_data, compute_std_for_matching_rows, save_matrix_to_csv, save_results_to_csv

parser = argparse.ArgumentParser(description="PyTorch cnn model")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default=".",
    help="path to dataset (default: actual directory)",
)
parser.add_argument(
    "--epochs", default=25, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "-b",
    "--batch-size",
    default=32,
    type=int,
    metavar="N",
    help="mini_batch size (default: 32), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.01,
    type=float,
    metavar="W",
    help="weight decay (default: 0.01)",
    dest="weight_decay",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--optimizer",
    default="Adam",
    help="choose the optimizer between SGD and Adam",
    type=str,
)
parser.add_argument(
    "--repetition",
    default=None,
    type=int,
    help="Choose the number of repetitions for the trainloop",
)
parser.add_argument(
    "--model", default="cnn", type=str, help="Choose the model between linear and cnn"
)

def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    """
    Combine all the function to train the model and save the results
    """
    args = parser.parse_args()
    if not os.path.exists("result"):
        os.makedirs("result")
    if not os.path.exists(os.path.join("result", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}")):
        os.makedirs(os.path.join("result", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"))

    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f"cuda:{args.gpu}")
            print(f"Use GPU: {args.gpu} for training")
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        print("using CPU, this will be slow")

    # define loss function (criterion), optimizer, and learning rate scheduler

    criterion = nn.BCEWithLogitsLoss().to(device)

    # data_loading code
    traindir = os.path.join(args.data, "train/image_list.txt")
    testdir = os.path.join(args.data, "test/image_list.txt")
    valdir = os.path.join(args.data, "validation/image_list.txt")
    trainloader, valloader, testloader = dataset_load(args, traindir, testdir, valdir)
    train_loss_matrix, train_accuracy_matrix, val_loss_matrix, val_accuracy_matrix = (
        [],
        [],
        [],
        [],
    )

    for r in range(args.repetition):
        np.random.seed(r)
        torch.manual_seed(r)
        random.seed(r)
        model = {
            "cnn": Predictor().to(device),
            "linear": LinearClassifier().to(device),
        }[args.model]
        model.reset_weights()
        optimizer = {
            "Adam": torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            ),
            "SGD": torch.optim.SGD(
                model.parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            ),
        }[args.optimizer]
        # Sets the learning rate to the initial LR decayed by 10 every 30 epochs
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        print(f"Repetition:{r+1}")
        list_train_loss, list_train_accuracy, list_val_loss, list_val_accuracy = train(
            modele=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            trainloader=trainloader,
            valloader=valloader,
            n_epoch=args.epochs,
            sched=scheduler,
            rep =r,
            args = args
        )

        train_loss_matrix.append(list_train_loss)
        train_accuracy_matrix.append(list_train_accuracy)
        val_loss_matrix.append(list_val_loss)
        val_accuracy_matrix.append(list_val_accuracy)

    std_train_loss = np.std(
        train_loss_matrix, axis=0
    )  # Compute std along the specified axis.
    std_train_accuracy = np.std(train_accuracy_matrix, axis=0)
    std_val_loss = np.std(val_loss_matrix, axis=0)
    std_val_accuracy = np.std(val_accuracy_matrix, axis=0)

    save_results_to_csv(
        filename="std_loss_and_accuracy",
        train_loss=std_train_loss,
        train_accuracy=std_train_accuracy,
        val_loss=std_val_loss,
        val_accuracy=std_val_accuracy,
        args=args
    )
    save_matrix_to_csv(
        filename="train_loss_per_repetition",
        matrix=train_loss_matrix,
        num_repetitions=args.repetition,
        args=args
    )
    save_matrix_to_csv(
        filename="val_loss_per_repetition",
        matrix=val_loss_matrix,
        num_repetitions=args.repetition,
        args=args
    )
    save_matrix_to_csv(
        filename="train_accuracy_per_repetition",
        matrix=train_accuracy_matrix,
        num_repetitions=args.repetition,
        args=args
    )
    save_matrix_to_csv(
        filename="val_accuracy_per_repetition",
        matrix=val_accuracy_matrix,
        num_repetitions=args.repetition,
        args=args
    )
    averages_train_loss = [
        sum(column) / len(column) for column in zip(*train_loss_matrix)
    ]
    averages_train_accuracy = [
        sum(column) / len(column) for column in zip(*train_accuracy_matrix)
    ]
    averages_val_loss = [sum(column) / len(column) for column in zip(*val_loss_matrix)]
    averages_val_accuracy = [
        sum(column) / len(column) for column in zip(*val_accuracy_matrix)
    ]
    save_results_to_csv(
        filename="average_loss_and_accuracy",
        train_loss=averages_train_loss,
        train_accuracy=averages_train_accuracy,
        val_loss=averages_val_loss,
        val_accuracy=averages_val_accuracy,
        args=args
    )


def dataset_load(args, train_path, test_path, val_path):
    """
    Dataset loader

    Parameters:
    --------------------------------
    train_path : str
        paths for each image in the train set
    test_path : str
        paths for each image in the test set
    val_path : str
        paths for each image in the validation set
    Return:
    --------------------------------
    trainloader : dataloader
        return the trainloader for the train set
    valloader: dataloader
        return the valloader for the validation set
    testloader:
        return the testloader for the test set
    """
    with open(train_path, "r") as f:
        lines = f.readlines()  # [next(f) for _ in range(1000)]

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths = []
    labels = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(" ")
        if line[1] == "0" or line[1] == "1":
            image_paths.append("train/" + line[0])

            labels.append(int(line[1]))

    # Appliquer les transformations d'image si nécessaire
    transform_train = transforms.Compose(
        [
            ConvertToRGB(),
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with open(val_path, "r") as f:
        lines = f.readlines()  # [next(f) for _ in range(1000)]

        # Créer des listes pour stocker les chemins d'image et les étiquettes
        image_paths_val = []
        labels_val = []

        # Parcourir chaque ligne du fichier texte et extraire les informations
        for line in lines:
            line = line.strip().split(" ")
            if line[1] == "0" or line[1] == "1":
                image_paths_val.append("validation/" + line[0])

                labels_val.append(int(line[1]))

        # Appliquer les transformations d'image si nécessaire
        transform_val = transforms.Compose(
            [
                ConvertToRGB(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        with open(test_path, "r") as f:
            lines = f.readlines()

        # Créer des listes pour stocker les chemins d'image et les étiquettes
        image_paths_test = []
        labels_test = []

        # Parcourir chaque ligne du fichier texte et extraire les informations
        for line in lines:
            line = line.strip().split(" ")
            image_paths_test.append("test/" + line[0])
            labels_test.append(int(line[1]))

        # Appliquer les transformations d'image si nécessaire
        transformation = transforms.Compose(
            [
                ConvertToRGB(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if args.model == "linear":
            transform_train.transforms.append(
                transforms.Lambda(lambda x: torch.flatten(x))
            )
            transform_val.transforms.append(
                transforms.Lambda(lambda x: torch.flatten(x))
            )
            transformation.transforms.append(
                transforms.Lambda(lambda x: torch.flatten(x))
            )

        df_val = CustomDataset(image_paths_val, labels_val, transform=transform_val)
        valloader = DataLoader(df_val, batch_size=args.batch_size, shuffle=True)

        df_test = CustomDataset(image_paths_test, labels_test, transform=transformation)
        testloader = DataLoader(df_test, batch_size=args.batch_size, shuffle=True)

        df_train = CustomDataset(image_paths, labels, transform=transform_train)
        trainloader = DataLoader(df_train, batch_size=args.batch_size, shuffle=True)

        return trainloader, valloader, testloader


def train(modele, optimizer, criterion, device, trainloader, n_epoch, valloader, sched, args, rep):
    """
    A function that allows to train the model for one repetition

    Parameters:
    --------------------------------
    modele : LinearClassifier or Predictor
        model choosen for the train
    criterion : torch.nn.Module
        loss function
    device : torch.device
        The device that runs the model
    optimizer : torch.optim
        optimizer for the model
    trainloader : dataloader
        dataloader used for the train set
    n_epoch : int
        number of epoch for one repetition
    valloader : dataloader
        dataloader used for the validation set
    sched : StepLR()
        Scheduler used for the train function

    Return:
    ---------------------------------
    list_train_loss: list
        list with all value for the training loss for the train set
    list_train_accuracy:
        list with all value for the training accuracy for the train set
    list_val_loss:
        list with all value for the validation loss for the validation set
    list_val_accuracy:
        list with all value for the validation accuracy for the validation set
    """

    train_loss_0, train_accuracy_0 = iterate_data(
        model=modele,
        dataloader=trainloader,
        is_training=False,
        criterion=criterion,
        device=device,
    )
    val_loss_0, val_accuracy_0 = iterate_data(
        model=modele,
        dataloader=valloader,
        is_training=False,
        criterion=criterion,
        device=device,
    )

    print(
        f"Train Loss 0: {train_loss_0}, Train Accuracy 0 : {train_accuracy_0} Val Loss 0: {val_loss_0}, Val Accuracy 0: {val_accuracy_0}%"
    )

    list_train_loss = [train_loss_0]
    list_train_accuracy = [train_accuracy_0]
    list_val_loss = [val_loss_0]
    list_val_accuracy = [val_accuracy_0]
    best_accuracy = 0.0
    for epoch in range(n_epoch):
        # train
        train_loss, train_accuracy = iterate_data(
            model=modele,
            dataloader=trainloader,
            is_training=True,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=sched,
        )
        list_train_loss.append(train_loss)
        list_train_accuracy.append(train_accuracy)

        # validation
        val_loss, val_accuracy = iterate_data(
            model=modele,
            dataloader=valloader,
            is_training=False,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        list_val_loss.append(val_loss)
        list_val_accuracy.append(val_accuracy)

        sched.step()

        if max(list_val_accuracy) > best_accuracy:
            best_accuracy = max(list_val_accuracy)
            torch.save(modele.state_dict(), os.path.join(os.path.join("result", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), f"best_model_{rep}.pth"))

        print(
            f"Epoch [{epoch+1}/{n_epoch}], Train Loss: {train_loss}, Train Accuracy : {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%"
        )

    return list_train_loss, list_train_accuracy, list_val_loss, list_val_accuracy

if __name__ == "__main__":
    main()