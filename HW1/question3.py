
#!/usr/bin/env python

import argparse
import random
import os
from itertools import count
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import torchvision
from torchvision import transforms
from tqdm import tqdm
from glob import glob


from sklearn.metrics import confusion_matrix, f1_score

import seaborn as sn
import pandas as pd
import itertools


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

  
transform_gray = transforms.Compose([
    # transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])  


## Question 3.1
class LogisticRegression(nn.Module):

    def __init__(self, n_classes, n_features, **kwargs):
        """
        n_classes (int)
        n_features (int)
        """
        super(LogisticRegression, self).__init__()
        self.layer = nn.Linear(n_features, n_classes)
        # nn.CrossEntropy already computes softmax.

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return self.layer(x)


## Question 3.2

class FeedforwardNetwork(nn.Module):
    def __init__(
            self, n_classes, n_features, hidden_sizes, layers,
            activation_type, dropout, **kwargs):
        """
        n_classes (int)
        n_features (int)
        hidden_sizes (list) Note: can also be a int
        activation_type (str)
        dropout (float): dropout probability
        """
        super(FeedforwardNetwork, self).__init__()

        print(f"Dropout: {dropout}")
        print(f"Hidden sizes: {hidden_sizes}")


        assert len(hidden_sizes) == layers
        layer_dimensions = [n_features] + hidden_sizes + [n_classes]
        sizes = [[layer_dimensions[i], layer_dimensions[i+1]] for i in range(len(layer_dimensions)-1)]
        
        
        activation = nn.ReLU() if activation_type == "relu" else nn.Tanh()
        dropout_layer = nn.Dropout(dropout)

        modules = []
        for s_in, s_out in sizes[:-1]:
            modules.append(nn.Linear(s_in, s_out))
            modules.append(activation)
            modules.append(dropout_layer)
        
        # Crossentropy already computes logsoftmax; also no dropout on output layer
        modules.append(nn.Linear(sizes[-1][0], sizes[-1][1]))


        self.ffnn = nn.Sequential(*modules)

        

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return self.ffnn(x)


## Question 3.3
class CNN(nn.Module):
    def __init__(self,n_classes, n_features, dropout=0.0, channels=1, kernel_size=2, layer_1_output_channel=8, layer_2_output_channel=16):
        """
        num_classes (int)
        n_features (int) Note: 2d representation
        channels (int) Note: set to 3 for RGB and 1 for grayscale
        activation_type (str)
        dropout (float): dropout probability
        kerner_size (int): kernel size 
        """
        super().__init__()  

        print(f"Dropout: {dropout}")
        print(f"Channels: {channels}")
        print(f"Kernel Size: {kernel_size}")
        print(f"C1 Out Channel: {layer_1_output_channel}")
        print(f"C2 Out Channel: {layer_2_output_channel}")

        self.convolutions = nn.Sequential(
            nn.Conv2d(channels,layer_1_output_channel,kernel_size=kernel_size,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(layer_1_output_channel,layer_2_output_channel,kernel_size=kernel_size,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout(dropout),
            nn.Flatten()
        )

        linear_in_features = self.convolutions(torch.empty(1, channels , n_features, n_features)).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=linear_in_features, out_features=1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=1024,out_features=n_classes),
            nn.ReLU()
        )
        
        
    def forward(self,input):   
        """
        x (batch_size x channels x n_features x n_features): a batch of training examples
        """
        return self.classifier(self.convolutions(input))


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    # adapted from pytorch class notebook
    # clear the gradients
    optimizer.zero_grad()
    # compute the model output
    yhat = model(X)
    # calculate loss
    loss = criterion(yhat, y)
    # credit assignment
    loss.backward()
    # update model weights
    optimizer.step()

    return loss


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels, scores


def evaluate(model, dataloader, confusion=False, modeltype='cnn'):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    n_correct=0
    n_possible=0
    y_pred = []
    y_true = []
    model.eval()
    losses = []

    with torch.no_grad():

        for X, y in dataloader:
            if modeltype!='cnn':
                X = X.reshape(X.shape[0],-1)
            y_hat, dist = predict(model, X)
            y_pred.extend(y_hat)
            y_true.extend(y)

            loss = nn.CrossEntropyLoss()(dist, y)

            n_correct += (y == y_hat).sum().item()
            n_possible += float(y.shape[0])
            
            losses.append(loss)
            
        f1 = f1_score(y_true, y_pred, average="macro")

    if confusion:
        classes = ('T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig('confusion_matrix.png')
    model.train()
   
    return n_correct / n_possible, torch.tensor(losses).mean().item(), f1


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')



#### To be used if you want to visualise the activations of the CNN convolutions

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_feature_maps_conv1(model, example, epoch):
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    data, _ = example
    data.unsqueeze_(0)
    _ = model(data)

    plt.imshow(data.reshape(28,-1)) 
    plt.savefig('original_image.pdf')

    k=0
    act = activation['conv1'].squeeze()
    
    fig,ax = plt.subplots(2,4,figsize=(12, 8))
    for i in range(act.size(0)//4):
        for j in range(4):
            ax[i,j].imshow(act[k].detach().cpu().numpy())
            k+=1  
            plt.savefig(str(epoch)+'_activation_maps1.pdf') 


def plot_feature_maps_conv2(model, example, epoch):
    
    model.conv2.register_forward_hook(get_activation('conv2'))
    
    data, _ = example
    data.unsqueeze_(0)
    _ = model(data)

    plt.imshow(data.reshape(28,-1)) 
    plt.savefig('original_image.pdf')

    k=0
    act = activation['conv2'].squeeze()
    fig,ax = plt.subplots(4,4,figsize=(12, 8))
    for i in range(act.size(0)//4):
        for j in range(act.size(0)//4):
            activ  = act[k].detach().cpu().numpy()
            ax[i,j].imshow(activ)
            k+=1  
            plt.savefig(str(epoch)+'_activation_maps2.pdf') 

##################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['logistic_regression', 'mlp','cnn'],
                        help="Which model should the script run?")
    parser.add_argument('-data', default='letter.data',
                        help="Path to letter.data OCR corpus.")
    parser.add_argument('-epochs', default=10, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=4, type=int,
                        help="Size of training batch.")
    parser.add_argument('-hidden_sizes', nargs='*', type=int, default=[200,20])
    parser.add_argument('-layers', type=int, default=2)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-activation',
                        choices=['tanh', 'relu'], default='relu')
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-bias', action='store_true',
                        help="""Whether to add an extra bias feature to all
                        samples in the dataset. In an MLP, where there can be
                        biases for each neuron, adding a bias feature to the
                        input is not sufficient.""")
    opt = parser.parse_args()

    configure_seed(seed=42)
    
    dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
    valid_ratio = 0.2  # Split to use 20% for development set

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                            train=True,
                                            transform= transform_gray, 
                                            download=True)
    
    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid =  int(valid_ratio * len(train_valid_dataset))
    train_dataset, dev_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                    transform= transform_gray, #transforms.ToTensor(),
                                                    train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=opt.batch_size)
    test_dataloader = DataLoader(
        test_dataset, batch_size=opt.batch_size)
    n_classes = len(set(train_dataset.dataset.targets.numpy()))
    n_feats = train_dataset.dataset.data.shape[1]
    print(n_feats)
    print(n_classes)

    # initialize the models
    if opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats*n_feats)
    elif opt.model == 'cnn':
        model = CNN(n_classes, n_feats,dropout=0.5, channels=1, kernel_size=3, layer_1_output_channel=16, layer_2_output_channel=32)
    else:#MLP


        model = FeedforwardNetwork(
            n_classes, n_feats*n_feats, opt.hidden_sizes, opt.layers,
            opt.activation, opt.dropout)


    # get an optimizer
    optims = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay)

    # get a loss criterion
    criterion = nn.CrossEntropyLoss()

    # training loop
    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in tqdm(train_dataloader):
            if not opt.model=='cnn':
                X_batch = X_batch.reshape(opt.batch_size,-1)     
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accuracy, valid_loss, _ = evaluate(model, dev_dataloader, confusion=False, modeltype=opt.model)
        print('Valid acc: %.4f' % (valid_accuracy))
        if len(valid_accs)<1 or valid_accuracy>max(valid_accs):
            path_to_model = opt.model+'_epoch_'+str(ii)+'.pt'
            torch.save({
                'epoch': ii,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': mean_loss,
                }, path_to_model)
        valid_accs.append(valid_accuracy)
        valid_losses.append(valid_loss)
        mean_loss = torch.tensor(train_losses).mean().item()


    ## load best model
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 

    test_acc, test_loss, test_f1 = evaluate(model, test_dataloader, confusion=True, modeltype=opt.model)
    print('Final Test acc: %.4f' % test_acc)
    print('Final Test f1: %.4f' % test_f1)
    # plot
    plot(epochs, train_mean_losses, ylabel='Train Loss', name='{}-training-loss-{}-{}-{}-{}-{}'.format(opt.model, opt.learning_rate, opt.hidden_sizes[0], opt.dropout, opt.activation, opt.optimizer))
    plot(epochs, valid_losses, ylabel='Validation Loss', name='{}-validation-loss-{}-{}-{}-{}-{}'.format(opt.model, opt.learning_rate, opt.hidden_sizes[0], opt.dropout, opt.activation, opt.optimizer))
    plot(epochs, valid_accs, ylabel='Validation Accuracy', name='{}-validation-accuracy-{}-{}-{}-{}-{}'.format(opt.model, opt.learning_rate, opt.hidden_sizes[0], opt.dropout, opt.activation, opt.optimizer))




if __name__ == '__main__':
    main()

# Reproducing results: 
# Logistic Regression: python hw1-q3-2023-implemented-fashion.py logistic_regression -epochs 15 -learning_rate 0.0005 -batch_size 1
# MLP: python hw1-q3-2023-implemented-fashion.py mlp -epochs 15 -batch_size 1 -learning_rate 0.001 -dropout 0.2 -hidden_sizes 200 130
# CNN: python hw1-q3-2023-implemented-fashion.py cnn -epochs 15 -batch_size 4 -learning_rate 0.01
