import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from kornia.filters import gaussian_blur2d

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Encoder(nn.Module):
# Keypoint encoder which has a hourglass-like structure
# with two downsampling layers and two upsampling layers
    def __init__(self, config, dataset_stats):
        super().__init__()
        self.num_feature_maps=config["num_keypoints"]
        self.width=config["width"]
        self.height=config["height"]
        self.channels=config["channels"]
        self.batch_norm=config["batch_norm"]
        self.padding=config["padding"]
        self.activation_score_threshold=config["activation_score_threshold"]
        self.fm_conv_1=nn.Conv2d(self.channels,self.num_feature_maps*2,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_1.weight)
        self.bn1=nn.BatchNorm2d(self.num_feature_maps*2)
        self.fm_conv_2=nn.Conv2d(self.num_feature_maps*2,self.num_feature_maps,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_2.weight)
        self.bn2=nn.BatchNorm2d(self.num_feature_maps)
        self.fm_conv_3=nn.ConvTranspose2d(self.num_feature_maps,self.num_feature_maps*2,kernel_size=3,stride=2)
        torch.nn.init.normal_(self.fm_conv_3.weight)
        self.bn3=nn.BatchNorm2d(self.num_feature_maps*2)
        self.fm_conv_4=nn.ConvTranspose2d(self.num_feature_maps*2,self.num_feature_maps,kernel_size=3+2*self.padding+1,stride=2)
        torch.nn.init.normal_(self.fm_conv_4.weight)
        self.bn4=nn.BatchNorm2d(self.num_feature_maps)
        # mean=[0.485, 0.456, 0.406]+[0.4]*(self.channels-3)
        # std=[0.229, 0.224, 0.225]+[0.225]*(self.channels-3)
        self.normalize = transforms.Normalize(mean=dataset_stats[0], std=dataset_stats[1])
        self.count=0

    def spatial_soft_argmax(self,x):
        N,KP,H,W=x.shape
        x=x.view(N,KP,H*W)
        # compute the softmax with the subtraction trick
        # subtract the maximum value doens't change the output of the softmax
        # good for numerical stability
        exp=torch.exp(x-torch.max(x,dim=-1,keepdim=True)[0])
        # the weights are the softmax of the feature maps
        weights=exp/(torch.sum(exp,dim=-1,keepdim=True)+1e-8)
        # create meshgrid of the pixel coordinates
        y_grid,x_grid=torch.meshgrid([torch.arange(0,H),torch.arange(0,W)])
        # move the meshgrid to the device
        x_grid=x_grid.flatten().to(device)
        y_grid=y_grid.flatten().to(device)
        # compute the expected coordinates of the softmax
        # the expected values of the coordinates are the weighted sum of the softmax
        expected_x=torch.sum(weights*x_grid,dim=-1,keepdim=True)
        expected_y=torch.sum(weights*y_grid,dim=-1,keepdim=True)
        # concatenate the expected coordinates to the feature maps
        # subtract padding to get the coordinates in the original image
        if self.padding > 0: 
            coords=torch.cat([expected_x-self.padding,expected_y-self.padding],dim=-1)
        else:
            coords=torch.cat([expected_x,expected_y],dim=-1)
        # return the expected coordinates
        return coords

    def forward(self, x):
        # N batch_size, SF number of stucked frames, C channels, H height, W width
        N,SF,C,H,W=x.shape
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
        # second convolution with leaky relu and batch norm
        # input channels = 2*KP, output channels = KP
        # N * SF x KP x H/4 x W/4
        x=self.fm_conv_2(x)
        x=F.leaky_relu(x)
        if self.batch_norm:
            x=self.bn2(x)
        # first deconvolution with leaky relu and batch norm
        # input channels = KP, output channels = 2*KP
        # N * SF x 2*KP x H/2 x W/2
        x=self.fm_conv_3(x)
        x=F.leaky_relu(x)
        if self.batch_norm:
            x=self.bn3(x)
        # second deconvolution with leaky relu and batch norm
        # input channels = 2*KP, output channels = KP
        # N * SF x KP x H+padding x W+padding
        x=self.fm_conv_4(x)
        x=F.relu(x)
        if self.batch_norm:
            x=self.bn4(x)
        # compute the activation_score as the maximum of feature maps
        # N * Sf x KP
        status_activation_score=torch.amax(x,dim=(-1,-2))
        # log the activation score as points to wandb
        wandb.log({'activation_score_{0}'.format(i):status_activation_score[0,i].item() for i in range(self.num_feature_maps)})
        # threshold the activation score
        active_status=torch.sigmoid(1000*(status_activation_score-self.activation_score_threshold)).unsqueeze(-1)
        # compute the coordinates of the keypoints
        coords=self.spatial_soft_argmax(x)
        # concatienate the order, the status and the coordinates
        # N * SF x KP x 3
        kp=torch.cat([coords,active_status],dim=-1)
        # reshape the coordinates
        # N x SF x KP x 3
        kp=kp.view(N,SF,-1,3)
        # if self.count%500==0:
        #     self.log_feature_maps(x)
        self.count+=1
        return kp

    def log_feature_maps(self,x):
        N,KP,H,W=x.shape
        # create a figure and a grid of subplots for KP feature maps
        s=int(np.sqrt(KP))
        fig,ax=plt.subplots(s,KP//s, sharex=True, sharey=True)
        # iterate over the KP feature maps
        for i in range(KP):
            # select the i-th feature map
            im=ax[i%s,i//s].imshow(x[0,i,:,:].detach().cpu().numpy(), cmap='jet')
        # add the title
        fig.suptitle("Feature maps")
        # log the figure to wandb
        wandb.log({"Feature Maps":wandb.Image(fig)})
        # close the figure
        plt.close(fig)