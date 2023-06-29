import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='PyTorch linear model')
parser.add_argument('data', metavar='DIR', nargs='?', default='.',
                    help='path to dataset (default: actual directory)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--optimizer', default='SGD', 
                    help = 'choose the optimiizer between SGD and ADAM', type = str)

class AlphaChannelRemoval(object):
    def __call__(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
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
        label = F.one_hot(torch.tensor(label), num_classes=12).squeeze().to(torch.float32)
        return image, label

def main():
    if not os.path.exists('result'):
        os.makedirs('result')

def main_worker(gpu, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model


    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
            else:
                model.cuda()

    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # define loss function (criterion), optimizer, and learning rate scheduler

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    else :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                     weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    

    #data_loading code
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    valdir = os.path.join(args.data, 'validation')
    valloader, testloader, trainloader = dataset_load(traindir, testdir, valdir)
    


def dataset_load(args, train_path, test_path, val_path):

    with open(train_path, 'r') as f:
        lines = f.readlines()

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths = []
    labels = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(' ')
        image_paths.append("train/" + line[0])
        labels.append(int(line[1]))

    # Appliquer les transformations d'image si nécessaire
    transform = transforms.Compose([
        AlphaChannelRemoval(),
        #ResizeWithBackground((384,216)),
        transforms.ToTensor(),
        transforms.Normalize((0.8869, 0.8856, 0.8831),(0.1999, 0.2032, 0.2101)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])
    df_train = CustomDataset(image_paths, labels, transform=transform)
    trainloader = DataLoader(df_train, batch_size= args.batch_size)

    with open(test_path, 'r') as f:
        lines = f.readlines()

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths = []
    labels = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(' ')
        image_paths.append("test/" + line[0])
        labels.append(int(line[1]))

        df_test = CustomDataset(image_paths, labels, transform=transform)
        testloader = DataLoader(df_test, batch_size=args.batch_size, shuffle=True)

    with open(val_path, 'r') as f:
        lines = f.readlines()

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths = []
    labels = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(' ')
        image_paths.append("val/" + line[0])
        labels.append(int(line[1]))

        df_val = CustomDataset(image_paths, labels, transform=transform)
        valloader = DataLoader(df_test, batch_size=args.batch_size, shuffle=True)
    return valloader, testloader, trainloader

def train(model, optimizer, criterion, device, trainloader, n_epoch, valloader):

    model.train()

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    all_loss = []
    for epoch in range(n_epoch):
        temp_loss = []
        for images, labels in trainloader:
            images = images.to(device, non_blocking = True) 
            labels = labels.to(device, non_blocking = True) 
            # Réinitialiser les gradients
            optimizer.zero_grad()
            # Calculer les sorties du modèle
            output = model(images)
            # Calculer la perte en utilisant les sorties et les étiquettes
            loss = criterion(output, labels)
            # Rétropropagation et mise à jour des poids
            loss.backward()
            optimizer.step()

            temp_loss.append(loss.item())

        scheduler.step()

        train_loss = torch.mean(torch.tensor(temp_loss), dim=0).item()
        all_loss.append(train_loss)

            # Boucle de validation
        model.eval()  # Mode d'évaluation
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in valloader:
                # Préparation des données de validation
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                val_loss += criterion(output, labels).item()

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calcul de la précision et de la perte moyenne de validation pour cette époque
        val_accuracy = 100 * correct / total
        val_loss /= len(valloader)
        with open('output.txt', 'w') as f:
            print(f"Epoch [{epoch+1}/{n_epoch}], Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")
            print(f"Epoch: {epoch}, loss: {np.mean(temp_loss)}", file=f)

        plt.plot(all_loss)
        plt.savefig(os.path.join('result','loss_graph.png'))
    
    torch.save(model.state_dict(), os.path.join('result','linear_model.pth'))
        


if __name__ == '__main__':
    main()