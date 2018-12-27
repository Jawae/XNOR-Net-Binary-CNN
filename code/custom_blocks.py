"""
    Custom blocks for usage

    Version 2 2018/11/17 Abhinav Kumar    --- More modules added
    Version 1 2018/10/09 Abhinav Kumar
"""	


import torch
import torch.nn as nn
import math

################################################################################
# Calculates ReLU output for scaled linear variables
# f(x) =  ReLU(m*x + c)
# 
# Output is same shape as input  
################################################################################
class AffineReLU(torch.nn.Module):
    def __init__(self, m=1, c=0):
        super(AffineReLU, self).__init__()
        self.relu = torch.nn.ReLU()
        self.m = m
        self.c = c

    def forward(self, x):
        x = self.relu(self.m*x + self.c)
        
        return x

################################################################################        
# Calculates thresholding based on variable input
#
# f(x) =  1 for x >=b
#      =  0 for x < b
# 
# Output is same shape as input        
################################################################################
class Thresholder(nn.Module):
    def __init__(self, b=0, eps=1e-6):
        super(Thresholder, self).__init__()
        self.arelu1     = AffineReLU(c = -(b+eps))
        self.arelu2     = AffineReLU(m = -1000, c = 1)
        self.arelu3     = AffineReLU(m = -1000, c = 1)    

    def forward(self, x):        
        x = self.arelu1(x)
        x = self.arelu2(x)
        x = self.arelu3(x)        
        
        return x

################################################################################
# Calculates mean for each of the variable
# Output is same shape as input
################################################################################
class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
    
    def forward(self, x):
        s = x.shape
        
        # 2d conv layers output    
        if(len(s) == 4):
            mean = torch.mean(torch.mean(torch.mean(x,dim=2),dim=2),dim=1)
            mean.unsqueeze_(-1)
            mean = mean.expand(s[0:-2])
            mean.unsqueeze_(-1)
            mean = mean.expand(s[0:-1])
            mean.unsqueeze_(-1)  
        # 1d conv layers input    
        elif(len(s) == 2):
            mean = torch.mean(x,dim=1)
            mean.unsqueeze_(-1)
        else:
            mean = torch.mean(x,dim=0)
        
        # 1/n already in mean
        # Expand now
        mean = mean.expand(s)
        
        return mean 

################################################################################
# Calculates alpha for each of the variable
# Output is same shape as input
################################################################################
class Alpha(nn.Module):
    def __init__(self):
        super(Alpha, self).__init__()
        self.mean      = Mean()
         
    def forward(self, x):
        x = self.mean(torch.abs(x)) # mean over absolute values will give 1 norm
        
        return x 

################################################################################
# Calculates mean for each of the variable
# Output is same shape as no of output filters
################################################################################
class MeanVector(nn.Module):
    def __init__(self, axis = -1):
        super(MeanVector, self).__init__()
        # axis defines along which axis means to be taken
        self.axis = axis
    
    def forward(self, x):
        if(self.axis < 0):
            s = x.shape
            
            # 2d conv layers output    
            if(len(s) == 4):
                mean = torch.mean(torch.mean(torch.mean(x,dim=2),dim=2),dim=1)     
            # 1d conv layers input    
            elif(len(s) == 2):
                mean = torch.mean(x,dim=1)
            else:
                mean = torch.mean(x,dim=0)
        
        else:
            mean = torch.mean(x, dim = self.axis)
            
        # 1/n already in mean    
        return mean 


################################################################################
# Calculates alpha for each of the variable
# Output is same shape as no of output  filters
################################################################################
class AlphaVector(nn.Module):
    def __init__(self):
        super(AlphaVector, self).__init__()
        self.mean_vector      = MeanVector()
         
    def forward(self, x):
        x = self.mean_vector(torch.abs(x)) # mean over absolute values will give 1 norm
        
        return x 


################################################################################
# Calculates beta for each of the variable
# Output is same shape as no of output  filters
################################################################################
class BetaVector(nn.Module):
    def __init__(self):
        super(BetaVector, self).__init__()
        self.mean_vector      = MeanVector(axis = 1)
         
    def forward(self, x):
        x = self.mean_vector(torch.abs(x)) # mean over absolute values will give 1 norm
        
        return x 


################################################################################
# Binarises the weight matrix on the basis of sign function
# Output is same shape as input
################################################################################
class Signum(nn.Module):
    def __init__(self, b=0, eps=1e-6):
        super(Signum, self).__init__()
        self.mean       = Mean() 
        self.arelu1     = AffineReLU(c = -(b+eps))
        self.arelu2     = AffineReLU(m = -1000, c = 1)
        self.arelu3     = AffineReLU(m = -1000, c = 1)   

    def forward(self, x):        
        mean = self.mean(x)
        x = x - mean
        
        x = self.arelu1(x)
        x = self.arelu2(x)
        x = self.arelu3(x) # x in range [0,1)        
        x = 2*(x - 0.5)    # x in range [-1,1)
        
        return x

################################################################################
# Expands and multiplies two tensors x and y.
# x is assumed to be of higher dimensions        
################################################################################
class ExpandAndMultiply(nn.Module):
    def __init__(self):
        super(ExpandAndMultiply, self).__init__()
        
    def forward(self, x, y):
        s = x.shape
        t = y.shape
           
        if(len(s) == 4):
            
            # 2d conv layers output
            # Since this layer is used after conv layer
            # x = batch x channels x height x width
            # y =         channels
            # reshape y keeping batch fixed 
            if(len(t)==1):   
                x = x.permute(0,3,2,1)
                x = x*y
                x = x.permute(0,3,2,1)
            
            # 2d conv layers output multiplied by beta
            # Since this layer is used after conv layer
            # x = batch x channels x height x width
            # y = batch            x height x width
            # reshape y keeping batch fixed    
            elif(len(t)==3):
                diff1  = y.shape[1] - x.shape[2]
                shift1 = diff1/2
                diff2  = y.shape[2] - x.shape[3]
                shift2 = diff2/2
                y      = y[:, shift1: shift1+x.shape[2], shift2: shift2+x.shape[3]]
                
                x      = x.permute(1,0,2,3)
                x      = x*y
                x      = x.permute(1,0,2,3) 
                
        # 1d conv layers input
        # Since this layer is used after affine layer
        # x = batch x dim
        # y = dim
        # no need of permutation
        elif(len(s) == 2): 
            if (x.shape[1] == y.shape[0]):
                x = x*y
            else:
                x = x.permute(1,0)
                x = x*y
                x = x.permute(1,0)
            
        return x
        
################################################################################
# Implements an XNOR operation with Tensors. Assigns 1D or 2D convolution blocks
# depending on whether linear is True or False respectively.       
################################################################################        
class MultiplicationWithXNOR(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=-1, stride=0, padding=0, groups=1, dropout=0, Linear=False):
        super(MultiplicationWithXNOR, self).__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.dropout_p   = dropout
        self.linear_flag = Linear
        
        # This is for input
        self.signum1 = Signum()
        self.beta    = BetaVector()
        self.mul1    = ExpandAndMultiply()
        
        # This is for weights
        self.signum2 = Signum()
        self.alpha   = AlphaVector()
        self.mul2    = ExpandAndMultiply()
        
        # This is for dropout
        if self.dropout_p !=0:
            self.dropout = nn.Dropout(dropout)
        
        # Add suitable batch norm and conv layers
        if (self.linear_flag):
            self.bn     = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv   = nn.Linear(input_channels, output_channels)
        else:
            self.bn     = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv   = nn.Conv2d(input_channels, output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
                
        # Add an inplace ReLU at the end
        self.relu = nn.ReLU(inplace=True)
        
        
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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                if hasattr(m.weight, 'data'):
                    m.weight.data.zero_().add_(1.0)
        
    def forward(self, x, w, b):
        # Pass through batch norm
        x = self.bn(x)
        
        # Binarize inputs
        h    = self.signum1(x)
        beta = self.beta(x)
                
        # This is for dropout
        if self.dropout_p !=0:
            x = self.dropout(x)
        
        # Binarize weights and set biases to zero
        self.conv.weight.data = self.signum2(w.data)
        self.conv.bias.data   = torch.zeros(b.data.shape).type(b.data.type()) 
        alpha = self.alpha(w)
        
        
        # Convolve now
        x = self.conv(x)
        
        # Multiply by alpha and beta
        x = self.mul2(x, alpha)
        
        x = self.mul1(x, beta )
        
        x = self.relu(x)
        return x
