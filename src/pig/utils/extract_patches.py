import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchExtractor(nn.Module):
# a calss to extract patches around keypoints from a given image
    def __init__(self, config, std, aggregate=False, mask=True):
        super().__init__()
        # threshold for featuremap generation
        self.threshold=torch.nn.Threshold(-1*config['threshold_for_featuremaps'],1)
        # std for featuremap generation
        self.std_for_featuremap_generation=std
        # aggregate the featuremaps
        self.aggregate=aggregate
        self.mask=mask
    
    def forward(self, coords, size):
        N,SF,KP,_=coords.shape
        H,W=size
        std=torch.tensor(self.std_for_featuremap_generation).to(device)
        pi=torch.tensor(3.14159265358979323846).to(device)
        # pixel coordinates
        # x - N*KP x H x W x 1
        x = torch.arange(0,W,device=device).view(1,-1,1).repeat(N*SF*KP,1,H)
        # y - N*KP x H x W x 1
        y = torch.arange(0,H,device=device).view(1,1,-1).repeat(N*SF*KP,W,1)
        # pixels - N x H x W x 2
        pixels=torch.stack([x,y],dim=-1)
        # reshape the coordinates
        # N*KP x 2
        coords=coords.contiguous().view(N*SF*KP,2)
        # adjust the coordinates to a compatible shape
        coords=coords.view(N*SF*KP,1,1,2).repeat(1,W,H,1)
        # calculate the gaussian
        # squared_distances - N*KP x H x W x 1
        squared_distances=torch.sum((pixels-coords)**2,dim=-1)
        # # visualize the squared_distances
        # plt.imshow(squared_distances[0].detach().cpu().numpy(), cmap='jet')
        # plt.show()
        # exp_term - N*KP x H x W
        exp_term=torch.exp(-1*(squared_distances)/(2*std**2))
        # gaussian - N*KP x H x W 
        fm=exp_term/(2*std**2*torch.sqrt(2*pi))
        # # visualize the gaussian
        # plt.imshow(fm[0].detach().cpu().numpy(), cmap='gray')
        # plt.show()
        # reshape the fm
        # N x KP x H x W
        fm=fm.view(N,SF,KP,W,H)
        fm=fm.permute(0,1,2,4,3)
        if self.aggregate:
            fm=torch.sum(fm,dim=2)
        # threshold the fm
        fm=self.threshold(-fm)
        # # visualize the thresholded gaussian
        # plt.imshow(fm[0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        return fm
