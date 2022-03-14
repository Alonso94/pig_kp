import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class RepresentationModel(nn.Module):
# A representation model that takes a patch and return a representation
# that can be used to compute the loss
# it consists of a convolutional layer, and two linear layers
    def __init__(self, config):
        super().__init__()
        self.representation_size=config['representation_size']
        self.width=config['width']
        self.height=config['height']
        self.conv=nn.Conv2d(3,1,kernel_size=3,stride=1,padding=1)
        torch.nn.init.uniform_(self.conv.weight)
        self.linear_1=nn.Linear(self.width*self.height,2*self.representation_size)
        torch.nn.init.uniform_(self.linear_1.weight)
        self.linear_2=nn.Linear(2*self.representation_size,self.representation_size)
        torch.nn.init.uniform_(self.linear_2.weight)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    def forward(self, x):
        N,SF,KP,C,H,W=x.shape
        # normalize
        x=self.normalize(x.float())
        # reshape to have a have a batch of images
        # N * SF x C x H xW
        x=x.view(-1,C,H,W)
        # first convolution with leaky relu and batch norm
        # input channels = C, output channels = 1
        # N * SF x 1 x H x W
        x=self.conv(x)
        x=F.leaky_relu(x)
        x=x.view(-1,self.width*self.height)
        # x.register_hook(lambda grad: print(grad.mean()))
        # x.register_hook(lambda grad: (grad * 10).float())
        # first linear layer with leaky relu
        # R is the representation size
        # input channels = 1, output channels = 2*R
        # N * SF x 2*R
        x=self.linear_1(x)
        x=F.leaky_relu(x)
        # x.register_hook(lambda grad: (grad * 10).float())
        # x.register_hook(lambda grad: print(grad.mean()))
        # second linear layer
        # input channels = 2*KP, output channels = R
        # N * SF x R
        x=self.linear_2(x)
        # # constrain the output to be positive using softplus
        x=F.softplus(x)
        # normalize the output to be between 0 and 1
        x=x/x.max()
        # reshape
        x=x.view(N,SF,KP,self.representation_size)
        # x.register_hook(lambda grad: (grad * 100).float())
        # x.register_hook(lambda grad: print(grad.mean()))
        return x
