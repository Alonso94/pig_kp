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

class RepresentationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.representation_size=config['representation_size']
        self.num_feature_maps=config["num_keypoints"]
        self.batch_norm=config["batch_norm"]
        self.width=config["width"]
        self.height=config["height"]
        self.channels=config["channels"]
        self.fm_conv_1=nn.Conv2d(self.channels,self.num_feature_maps*2,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_1.weight)
        self.bn1=nn.BatchNorm2d(self.num_feature_maps*2)
        self.fm_conv_2=nn.Conv2d(self.num_feature_maps*2,self.num_feature_maps,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_2.weight)
        self.bn2=nn.BatchNorm2d(self.num_feature_maps)
        self.linear_1=nn.Linear(int(self.num_feature_maps*(self.width/4-1)*(self.height/4-1)),2*self.representation_size)
        torch.nn.init.normal_(self.linear_1.weight)
        self.linear_2=nn.Linear(2*self.representation_size,self.representation_size)
        torch.nn.init.normal_(self.linear_2.weight)
        mean=[0.485, 0.456, 0.406]+[0.4]*(self.channels-3)
        std=[0.229, 0.224, 0.225]+[0.225]*(self.channels-3)
        self.normalize = transforms.Normalize(mean,std)

    def forward(self, x):
        # N batch_size, SF number of stucked frames, C channels, H height, W width
        N,SF,KP,C,H,W=x.shape
        # normalize
        x=self.normalize(x.float())
        # reshape to have a have a batch of images
        # N * SF x C x H xW
        x=x.view(-1,C,H,W)
        # first convolution with leaky relu and batch norm
        # input channels = C, output channels = 2*KP
        # N * SF x 2*KP x H/2 x W/2
        x=self.fm_conv_1(x)
        x=F.leaky_relu(x)
        if self.batch_norm:
            x=self.bn1(x)
        # x.register_hook(lambda grad: print("layer 1 out",grad.mean()))
        # grad mean = 1.08e-14
        # second convolution with leaky relu and batch norm
        # input channels = 2*KP, output channels = KP
        # N * SF x KP x H/4 x W/4
        x=self.fm_conv_2(x)
        x=F.leaky_relu(x)
        if self.batch_norm:
            x=self.bn2(x)
        x=x.view(N*SF*KP,-1)
        # x.register_hook(lambda grad: print("layer 2 out",grad.mean()))
        # grad mean = 1.08e-14
        # first linear layer with leaky relu
        # R is the representation size
        # input channels = KP, output channels = 2*R
        # N * SF x 2*R
        x=self.linear_1(x)
        x=F.leaky_relu(x)
        # x.register_hook(lambda grad: print("layer 3 out",grad.mean()))
        # grad mean = 1.08e-14
        # second linear layer
        # input channels = 2*KP, output channels = R
        # N * SF x R
        x=self.linear_2(x)
        x=F.relu(x)
        # reshape
        x=x.view(N,SF,KP,self.representation_size)
        # x.register_hook(lambda grad: print("layer 4 out",grad.mean()))
        # grad mean = 1.08e-14
        return x

class RepresentationDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.representation_size=config['representation_size']
        self.num_feature_maps=config["num_keypoints"]
        self.batch_norm=config["batch_norm"]
        self.width=config["width"]
        self.height=config["height"]
        self.channels=config["channels"]
        self.linear_3=nn.Linear(self.representation_size,2*self.representation_size)
        torch.nn.init.normal_(self.linear_3.weight)
        self.linear_4=nn.Linear(2*self.representation_size,int(self.num_feature_maps*(self.width/4-1)*(self.height/4-1)))
        torch.nn.init.normal_(self.linear_4.weight)
        self.fm_conv_3=nn.ConvTranspose2d(self.num_feature_maps,self.num_feature_maps*2,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_3.weight)
        self.bn3=nn.BatchNorm2d(self.num_feature_maps*2)
        self.fm_conv_4=nn.ConvTranspose2d(self.num_feature_maps*2,self.channels,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_4.weight)
        self.bn4=nn.BatchNorm2d(self.num_feature_maps)
        self.fm_conv_5=nn.ConvTranspose2d(self.channels,self.channels,kernel_size=2,stride=1)

    
    def forward(self, x):
        N,SF,KP,R=x.shape
        # reshape
        # N * SF x R
        x=x.view(-1,R)
        # first linear layer with leaky relu
        # input channels = R, output channels = 2*KP
        # N * SF x 2*KP
        x=self.linear_3(x)
        x=F.leaky_relu(x)
        # x.register_hook(lambda grad: print("layer 3 out",grad.mean()))
        # grad mean = 1.08e-14
        # second linear layer
        # input channels = 2*KP, output channels = KP
        # N * SF x KP
        x=self.linear_4(x)
        x=F.leaky_relu(x)
        # x.register_hook(lambda grad: print("layer 4 out",grad.mean()))
        # grad mean = 1.08e-14
        # reshape
        # N * SF x KP x H/4 x W/4
        x=x.view(N*SF*KP,self.num_feature_maps,int(self.height/4-1),int(self.width/4-1))
        # first convolution with leaky relu and batch norm
        # input channels = KP, output channels = 2*KP
        # N * SF x 2*KP x H/4 x W/4
        x=self.fm_conv_3(x)
        x=F.leaky_relu(x)
        if self.batch_norm:
            x=self.bn3(x)
        # x.register_hook(lambda grad: print("layer 5 out",grad.mean()))
        # grad mean = 1.08e-14
        # second convolution with leaky relu and batch norm
        # input channels = 2*KP, output channels = KP
        # N * SF x KP x H/2 x W/2
        x=self.fm_conv_4(x)
        x=F.leaky_relu(x)
        x=self.fm_conv_5(x)
        x=F.relu(x)
        # x.register_hook(lambda grad: print("layer 6 out",grad.mean()))
        # grad mean = 1.08e-14
        # reshape to N x SF x KP x C x H x W
        x=x.view(N,SF,KP,self.channels,self.height,self.width)
        return x