"""
    Used to visualise the weight of the filter
    
    Sample Run:
    python visualize_mnist_v5.py

"""	

from __future__ import print_function

#from PIL import Image
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import custom_blocks
import numpy as np
import math
import os

import sys

################################################################################
# Standard lenet_5 with modifications from
# https://arxiv.org/pdf/1603.05279.pdf
################################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First layer 
        self.conv1      = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.bn1        = nn.BatchNorm2d(num_features =20)
        self.relu1      = nn.ReLU()
        self.max1       = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # Second Layer
        self.conv2      = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.xnor2      = custom_blocks.MultiplicationWithXNOR(input_channels=20 , output_channels=50 , kernel_size=5, stride=1, padding=0, groups=1, dropout=0, Linear=False)        
        self.max2       = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC layer
        self.fc1        = nn.Linear(800, 500)
        self.xnor3      = custom_blocks.MultiplicationWithXNOR(input_channels=800, output_channels=500, kernel_size=5, stride=1, padding=0, groups=1, dropout=0, Linear=True) 
        
        # Final layer
        self.fc2_bn     = nn.BatchNorm1d(num_features=500)
        self.fc2_drop   = nn.Dropout(0.5)  
        self.fc2        = nn.Linear(500, 10)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Need to intialise weights and bias since not initialised by random
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # Weights already intialised by random values. Initialise bias
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
                m.bias.data.zero_()    
        
    def forward(self, x):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.clamp_(min=0.01)
        
        # Pass the info from the first layer
        x = self.max1(self.relu1(self.bn1(self.conv1(x))))
        
        # Second Layer
        x = self.xnor2(x, self.conv2.weight, self.conv2.bias)     
        x = self.max2(x)
        
        x = x.view(-1, 800)
        
        # FC layer
        x = self.xnor3(x, self.fc1.weight  , self.fc1.bias) 
        
        # Final layer 
        x = self.fc2(self.fc2_drop(self.fc2_bn(x)))
        
        return F.log_softmax(x, dim=1)
        

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
                                        
        optimizer.zero_grad()
        
        # compute output
        output      = model(data)
        
        # compute loss
        loss        = F.nll_loss(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                    
        if batch_idx % args.log_interval == 0:
            pass
            #print('Train Epoch {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
                        
            output     = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred       = output.max(1, keepdim=True)[1]                     # get the index of the max log-probability
            correct   += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test Epoch {}, loss {:.4f}, Accuracy {:.2f}%'.format(epoch, test_loss, acc))
    
    return acc

        
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
        
    parser.add_argument(      '--no-cuda'        , default= False               , action='store_true'                       , help='disables CUDA training')
    parser.add_argument(      '--seed'           , default= 1                   , type= int                                 , help='random seed (default: 1)')
    parser.add_argument('-a', '--augmentation'   , default= False                                                           , help='use augmentation (default: False)')   
    parser.add_argument(      '--std_dev'        , default= 0.05                                                            , help='if use augmentation, then standard value of noise (default: 0.05)')
    parser.add_argument('-b', '--binary_weight'  , default= False               , action= 'store_true'                      , help='use_binary_weights')
    
    # Optimizer arguments
    parser.add_argument(      '--batch-size'     , default= 128                 , type= int                                 , help='input batch size for training (default: 128)')
    parser.add_argument(      '--test-batch-size', default= 10000               , type= int                                 , help='input batch size for testing  (default: 10000)')
    parser.add_argument('-e', '--epochs'         , default= 60                  , type= int                                 , help='number of epochs to train     (default: 30)')    
    parser.add_argument(      '--lr'             , default= 0.01                , type= float                               , help='learning rate (default: 0.01)')
    parser.add_argument(      '--momentum'       , default= 0.9                 , type= float                               , help='SGD momentum  (default: 0.9)')
    parser.add_argument(      '--step_size'      , default= 30                  , type= int                                 , help='step-size     (defualt: 30)')
    parser.add_argument('-w', '--weight_decay'   , default= 0.00001             , type= float                               , help='weight-decay  (default: 0.00001)')
    parser.add_argument('-g', '--gamma'          , default= 0.1                 , type= float                               , help='gamma - factor by which learning rate decays (default: 0.1)')
    
    # Logging arguments
    parser.add_argument('-s', '--save_directory' , default= "models/lenet/run_5"                                            , help='how many epochs to wait before saving the model')
    parser.add_argument(      '--log-interval'   , default= 100                 , type= int                                 , help='how many batches to wait before logging training status')
    parser.add_argument(      '--test-interval'  , default= 1                   , type= int                                 , help='how many epochs to wait before testing on the val')    
    parser.add_argument(      '--save_interval'  , default= 20                  , type= int                                 , help='how many epochs to wait before saving the model')                                         
                    
    # Parse arguments now
    args = parser.parse_args()
    use_cuda = False#not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    transform_list=[ transforms.ToTensor()] # transforms.Normalize((0.1307,), (0.3081,))]
    
    model_name   = ''    
    model_folder = args.save_directory
    
        
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,               transform=transforms.Compose(transform_list)),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    if args.augmentation:
        stddev = args.std_dev
        print("Std deviation of the noise = " + str(stddev))
        distbn = torch.distributions.normal.Normal(0, stddev)
        transform_list.append(transforms.Lambda(lambda img: img + distbn.sample(img.shape)))
    else:
        print("No data augmentation with noise added in training")
            
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose(transform_list)),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    

    
    print("****************************************************************")
    print("\t\tLeNet on MNIST\t\t")
    print("****************************************************************")
    # Initialize the model
    model = Net().to(device)

    #print(model)
    model = torch.load("./models/lenet/run_13/checkpoint_14.pth")
    
    wt = model.xnor2.conv.weight.detach().numpy()
    print(wt.shape)

    fig, ax = plt.subplots(nrows=4, ncols=5)
    cnt = 0
    for row in ax:
        for col in row:
            col.imshow(wt[0,cnt,:,:], cmap="gray")
            col.axis('off')
            #col.spines['top'].set_visible(True)
            #col.spines['right'].set_visible(True)            
            #col.spines['left'].set_visible(True)
            #col.spines['bottom'].set_visible(True)
            cnt = cnt + 1
            #col.plot(x, y)
        #plt.imshow(wt[0,0,:,:], cmap="gray")
    fig.savefig('layer2.png')

                
if __name__ == '__main__':
    main()
