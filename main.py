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
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch linear model')
parser.add_argument('data', metavar='DIR', nargs='?', default='.',
                    help='path to dataset (default: actual directory)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--optimizer', default='Adam', 
                    help = 'choose the optimiizer between SGD and ADAM', type = str)

class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != 'RGB':
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
        label = F.one_hot(torch.tensor(label), num_classes=2).squeeze().to(torch.float32)
        return image, label

class LinearClassifier(torch.nn.Module):
  def __init__(self, input_dim=150528, output_dim=2):
    super(LinearClassifier, self).__init__()
    self.linear = torch.nn.Linear(input_dim, output_dim)
    #self.sigmoid = torch.nn.Sigmoid() 
    #self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
          x = self.linear(x)
          #x = rdm pour le modèle aléatoire
          #x = self.sigmoid(x)
          #x = self.softmax(x)
          return x

def main():
    args = parser.parse_args()
    if not os.path.exists('result'):
        os.makedirs('result')
    
    np.random.seed(69)
    torch.manual_seed(69)
    random.seed(69)
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model

    model = LinearClassifier()
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')

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

    criterion = nn.BCEWithLogitsLoss().to(device)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    
    else :
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                     weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    

    #data_loading code
    traindir = os.path.join(args.data, 'train/image_list.txt')
    testdir = os.path.join(args.data, 'test/image_list.txt')
    valdir = os.path.join(args.data, 'validation/image_list.txt')
    trainloader, valloader, testloader = dataset_load(args, traindir, testdir, valdir)
    train(modele = model,
          optimizer= optimizer,
          criterion = criterion,
          device= device,
          trainloader= trainloader,
          valloader= valloader,
          n_epoch= args.epochs)



def dataset_load(args, train_path, test_path, val_path):

    with open(train_path, 'r') as f:
        lines = f.readlines()#[next(f) for _ in range(1000)] 

    # Créer des listes pour stocker les chemins d'image et les étiquettes
    image_paths = []
    labels = []

    # Parcourir chaque ligne du fichier texte et extraire les informations
    for line in lines:
        line = line.strip().split(' ')
        if line[1] == "0" or line[1] == "1":
            
            image_paths.append("train/" + line[0])
        
            labels.append(int(line[1]))

    # Appliquer les transformations d'image si nécessaire
    transform_train = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    df_train = CustomDataset(image_paths, labels, transform=transform_train)
    trainloader = DataLoader(df_train, batch_size=args.batch_size, shuffle=True)

    with open(val_path, 'r') as f:
        lines = f.readlines()#[next(f) for _ in range(1000)] 

        # Créer des listes pour stocker les chemins d'image et les étiquettes
        image_paths = []
        labels = []

        # Parcourir chaque ligne du fichier texte et extraire les informations
        for line in lines:
            line = line.strip().split(' ')
            if line[1] == "0"  or line[1] == "1":
                
                image_paths.append("validation/" + line[0])
            
                labels.append(int(line[1]))

        # Appliquer les transformations d'image si nécessaire
        transform_val = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        df_val = CustomDataset(image_paths, labels, transform=transform_val)
        valloader = DataLoader(df_val, batch_size=args.batch_size, shuffle=True)

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

        # Appliquer les transformations d'image si nécessaire
        transformation = transforms.Compose([
            ConvertToRGB(),
            transforms.Resize(256), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        df_test = CustomDataset(image_paths, labels, transform=transformation)
        testloader = DataLoader(df_test, batch_size=args.batch_size, shuffle=True)
        return trainloader, valloader, testloader

def iterate_data(dataloader, model, criterion, device, optimizer=None, is_training=False):

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
            correct_predictions += (predicted == labels_).sum().item()  # Accumuler le nombre de prédictions correctes

            if is_training:
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                # update weights
                optimizer.step()

    mean_loss = torch.tensor(loss_list).mean().item()
    mean_accuracy = (100.0 * correct_predictions) / total_samples
    return mean_loss, mean_accuracy


def train(modele, optimizer, criterion, device, trainloader, n_epoch, valloader):

    list_train_loss = []
    list_train_accuracy = []
    list_val_loss = []
    list_val_accuracy = []

    for epoch in range(n_epoch):
        
        #train
        train_loss, train_accuracy = iterate_data(model = modele, 
                                                dataloader= trainloader, 
                                                is_training=True, 
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                device = device)
        list_train_loss.append(train_loss)
        list_train_accuracy.append(train_accuracy)


        #validation

        val_loss, val_accuracy = iterate_data(model = modele, 
                                                dataloader = valloader, 
                                            is_training= False,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            device = device)
        list_val_loss.append(val_loss)
        list_val_accuracy.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{n_epoch}], Train Loss: {train_loss}, Train Accuracy : {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")
   
        with open('output.txt', 'w') as f:
            print(f"Epoch [{epoch+1}/{n_epoch}], Train Loss: {train_loss}, Train Accuracy : {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}%")

    plot_graph(name = "Training loss of the linear model",
               object = list_train_loss, 
               label_one="train",
               name_f='train_loss')
    
    plot_graph(name = "Validation loss of the linear model",
               object=list_val_loss,
               label_one="val",
               name_f='val_loss')
    plot_graph(name= "Validation accuracy of the linear model",
               object=list_val_accuracy,
               label_one= "val",
               name_f='val_accuracy')
    
    torch.save(modele.state_dict(), os.path.join('result','linear_model.pth'))
        
def plot_graph(name,name_f, object, label_one, two_curves = False, second_name = None, second_label = None,ylabel_loss = True):
    plt.style.use("ggplot")
    plt.rcParams.update()
    plt.figure(figsize=(10, 5))
    plt.title(name)
    plt.plot(object, label=label_one)
    if two_curves == True:
        plt.plot(second_name, label=second_label)
    plt.xlabel("epochs")
    if ylabel_loss == True:
        plt.ylabel("loss")
    else:
        plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join('result',f'{name_f}.png'))

if __name__ == '__main__':
    main()