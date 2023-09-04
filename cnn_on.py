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


parser = argparse.ArgumentParser(description="PyTorch cnn model")
parser.add_argument(
    "data",
    metavar="DIR",
    nargs="?",
    default=".",
    help="path to dataset (default: actual directory)",
)
parser.add_argument(
    "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run"
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
    default=1024,
    type=int,
    metavar="N",
    help="mini_batch size (default: 1024), this is the total "
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

parser.add_argument(
    "--budget", default=1500, type=int, help="Choose the budget for the online transfer"
)
parser.add_argument(
    "--H", default=300, type=int, help="Choose the size for the buffer"
)
parser.add_argument(
    "--pretrained", help="Choose if you want a pretrained model", action='store_true'
)
parser.add_argument(
    "--random", help="Choose if you want a random distribution", action='store_true'
)


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

def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    """
    Combine all the function to train the model and save the results
    """
    args = parser.parse_args()
    if not os.path.exists("result_online"):
        os.makedirs("result_online")
    if not os.path.exists(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}")):
        os.makedirs(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"))

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
    testdir = os.path.join(args.data, "test/image_list.txt")
    online_loss_matrix, online_accuracy_matrix, test_loss_matrix, test_accuracy_matrix, db = (
        [],
        [],
        [],
        [],
        [],
    )
    print(args.pretrained)
    for r in range(args.repetition):
        np.random.seed(r)
        torch.manual_seed(r)
        random.seed(r)
        testloader, online_loader = dataset_load(args, testdir)
        model = {
            "cnn": Predictor().to(device),
            "linear": LinearClassifier().to(device),
        }[args.model]
        
        pre_path = os.path.join('result',f'{args.model}_{args.epochs}_{args.lr}_{args.optimizer}')
        path = os.path.join(pre_path,f'best_model_{r}.pth')
        print(args.pretrained)
        if args.pretrained == True:
            model.load_state_dict(torch.load(path))
            torch.save(model.state_dict(), os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), f"best_model_{r}.pth"))
            path_online = os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), f"best_model_{r}.pth")
        
        else:
            if r == 0:
                torch.save(model.state_dict(), os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), f"best_model.pth"))
                path_online = os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), f"best_model.pth")
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
        list_online_loss, list_online_accuracy, list_test_loss, list_test_accuracy, database = train(
            modele=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            online_loader=online_loader,
            testloader=testloader,
            n_epoch=args.epochs,
            sched=scheduler,
            budget = args.budget, 
            H = args.H, 
            path = path,
            args = args,
            rep = r,
            path_online = path_online
        )
        
        online_loss_matrix.append(list_online_loss)
        online_accuracy_matrix.append(list_online_accuracy)
        test_loss_matrix.append(list_test_loss)
        test_accuracy_matrix.append(list_test_accuracy)
        db.append(database)
        std_online_loss, std_online_accuracy, std_test_loss, std_test_accuracy = compute_std_for_matching_rows(
                                    online_loss_matrix, online_accuracy_matrix, test_loss_matrix, test_accuracy_matrix, axis=0
)

    with open(os.path.join(os.path.join("result_online", f"{args.model}_{args.epochs}_{args.lr}_{args.optimizer}"), 'database.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        for row in db:
            writer.writerow(row)
    save_results_to_csv(
        filename="std_loss_and_accuracy",
        online_loss=std_online_loss,
        online_accuracy=std_online_accuracy,
        test_loss=std_test_loss,
        test_accuracy=std_test_accuracy,
        args=args
    )
    save_matrix_to_csv(
        filename="online_loss_per_repetition",
        matrix=online_loss_matrix,
        num_repetitions=args.repetition,
        args= args
    )
    save_matrix_to_csv(
        filename="test_loss_per_repetition",
        matrix=test_loss_matrix,
        num_repetitions=args.repetition,
        args= args
    )
    save_matrix_to_csv(
        filename="online_accuracy_per_repetition",
        matrix=online_accuracy_matrix,
        num_repetitions=args.repetition,
        args= args
    )
    save_matrix_to_csv(
        filename="test_accuracy_per_repetition",
        matrix=test_accuracy_matrix,
        num_repetitions=args.repetition,
        args= args
    )
    averages_online_loss = [
        sum(column) / len(column) for column in zip(*online_loss_matrix)
    ]
    averages_online_accuracy = [
        sum(column) / len(column) for column in zip(*online_accuracy_matrix)
    ]
    averages_test_loss = [sum(column) / len(column) for column in zip(*test_loss_matrix)]
    averages_test_accuracy = [
        sum(column) / len(column) for column in zip(*test_accuracy_matrix)
    ]
    save_results_to_csv(
        filename="average_loss_and_accuracy",
        online_loss=averages_online_loss,
        online_accuracy=averages_online_accuracy,
        test_loss=averages_test_loss,
        test_accuracy=averages_test_accuracy,
        args= args
    )


def dataset_load(args, test_path):
    """
    Dataset loader

    Parameters:
    --------------------------------
    test_path : str
        paths for each image in the test set
    Return:
    --------------------------------
    testloader:
        return the testloader for the test set
    """
    
    with open(test_path, "r") as f:
        lines = f.readlines()

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths_test = []
    labels_test = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(" ")
        if line[1] == "0" or line[1] == "1":
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
        transformation.transforms.append(
            transforms.Lambda(lambda x: torch.flatten(x))
        )

    df_test = CustomDataset(image_paths_test, labels_test, transform=transformation)
    test_dataset_size = len(df_test)
    split_sizes = [test_dataset_size // 2, test_dataset_size - test_dataset_size // 2]
    test_online, test_dataset = random_split(df_test, split_sizes)
    

    desired_online_images = 1500
    desired_online_per_class = desired_online_images // 2  # 1250 pour chaque classe

    labels_one_hot = [df_test[i][1] for i in range(len(df_test))]
    labels_classes = [torch.argmax(label).item() for label in labels_one_hot]

    indices_class_0 = [i for i, label in enumerate(labels_classes) if label == 0]
    indices_class_1 = [i for i, label in enumerate(labels_classes) if label == 1]

    # Obtenez les indices pour test_online en prenant le min entre la taille réelle de la classe et le nombre souhaité
    split_point_0 = min(desired_online_per_class, len(indices_class_0))
    split_point_1 = min(desired_online_per_class, len(indices_class_1))

    indices_online = indices_class_0[:split_point_0] + indices_class_1[:split_point_1]
    indices_dataset = indices_class_0[split_point_0:] + indices_class_1[split_point_1:]

    test_online = Subset(df_test, indices_online)
    test_dataset = Subset(df_test, indices_dataset)

    online_loader = DataLoader(test_online, batch_size = 1, shuffle=args.random)
    testloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)

    return testloader, online_loader


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


def train(modele, optimizer, criterion, device, online_loader, n_epoch, testloader, sched, budget, H, path, args, rep, path_online):
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
    online_loader : dataloader
        dataloader used for the train set
    n_epoch : int
        number of epoch for one repetition
    testloader : dataloader
        dataloader used for the validation set
    sched : StepLR()
        Scheduler used for the train function

    Return:
    ---------------------------------
    list_online_loss: list
        list with all value for the training loss for the train set
    list_online_accuracy:
        list with all value for the training accuracy for the train set
    list_test_loss:
        list with all value for the validation loss for the validation set
    list_test_accuracy:
        list with all value for the validation accuracy for the validation set
    """



    strategy = Strategie(budget = budget, n = 2)
    strategy.reset()
    database = []
    list_online_loss = []
    list_online_accuracy = []
    list_test_loss = []
    list_test_accuracy = []
    db = []
    i = 0
    for inputs, labels in online_loader: 
        i+= 1
        print(i)
        inputs = inputs.to(device)
        output = modele(inputs)
        output = torch.argmax(output, dim=1)
        print(output)
        if strategy.get_action(output) == 1:

            labels = labels.squeeze(0)
            database.append((inputs.squeeze(0), labels))
            db.append(labels)
            strategy.update(output)
            
            if len(database) % H == 0 and len(database)<(strategy.budget + 1): #buffer
                #reentrainer le modele a partir de cette sauvegarde sur la bdd maj (avec une nouvelle obs)
                #Reset le model avec la sauvegarde pth
                ##########################################
                if args.pretrained == True:
                    modele.load_state_dict(torch.load(path))
                else:
                    modele.load_state_dict(torch.load(path_online))
                optimizer = {
                        "Adam": torch.optim.Adam(
                            modele.parameters(), lr=args.lr, weight_decay=args.weight_decay
                        ),
                        "SGD": torch.optim.SGD(
                            modele.parameters(),
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                        ),
                    }[args.optimizer]
                sched = StepLR(optimizer, step_size=10, gamma=0.1)
                selected_images_dataset = SelectedImagesDataset(database)
                selected_images_loader = DataLoader(selected_images_dataset, batch_size=args.batch_size, shuffle=True)
                best_accuracy = 0.0
                online_loss_0, online_accuracy_0 = iterate_data(
                    model=modele,
                    dataloader=online_loader,
                    is_training=False,
                    criterion=criterion,
                    device=device,
                )
                test_loss_0, test_accuracy_0 = iterate_data(
                    model=modele,
                    dataloader=testloader,
                    is_training=False,
                    criterion=criterion,
                    device=device,
                )

                print(
                    f"Train Loss 0: {online_loss_0}, Train Accuracy 0 : {online_accuracy_0} Val Loss 0: {test_loss_0}, Val Accuracy 0: {test_accuracy_0}%"
                )

                list_online_loss.append(online_loss_0)
                list_online_accuracy.append(online_accuracy_0)
                list_test_loss.append(test_loss_0)
                list_test_accuracy.append(test_accuracy_0)
                for epoch in range(n_epoch):
                    # train
                    online_loss, online_accuracy = iterate_data(
                        model=modele,
                        dataloader=selected_images_loader,
                        is_training=True,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        scheduler=sched,
                    )
                    list_online_loss.append(online_loss)
                    list_online_accuracy.append(online_accuracy)

                    # test
                    test_loss, test_accuracy = iterate_data(
                        model=modele,
                        dataloader=testloader,
                        is_training=False,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                    )
                    list_test_loss.append(test_loss)
                    list_test_accuracy.append(test_accuracy)
                    if args.pretrained == True:
                        if max(list_test_accuracy) > best_accuracy:
                            best_accuracy = max(list_test_accuracy)
                            torch.save(modele.state_dict(), path_online)
                    sched.step()
                    print(
                        f"Epoch [{epoch+1}/{n_epoch}], Online Loss: {online_loss}, Online Accuracy : {online_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%"
                    )

    return list_online_loss, list_online_accuracy, list_test_loss, list_test_accuracy, db


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

if __name__ == "__main__":
    main()

