from pig.losses.max import MaxMatchLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.morphology import dilation
from kornia.filters import sobel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500

import cv2
import numpy as np

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchContrastiveLoss(nn.Module):
    # extracting the represemtation vector for each keypoint, which consists of :
        # [ 
        #   0,1 - coordinates (x,y),
        #   2:2+n - appearance information (sum of pixel values in a ring around the keypoint),
        #   2+n:2+n+3 - appearance information (edges),
        #   
        # ]
    def __init__(self, config):
        super().__init__()
        # MCL loss
        self.mcl_loss=MaxMatchLoss(config)
    
    def forward(self, coords, feature_maps, images):
        # if self.number_of_samples is not None:
        #     # draw samples
        #     samples=torch.randint(0,self.num_keypoints,(self.number_of_samples,),device=device)
        #     # get the samples
        #     coords=coords[:,:,samples,:]
        #     feature_maps=feature_maps[:,:,samples,:]
        # coords.register_hook(lambda grad: print("coords",grad))
        # extract patches
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # # initialize the representation vector
        # # N*SF*KP x 2 (coordinates) + n_iterations + 3 (edges)
        # representation=torch.zeros((N,SF,KP,2),device=device)
        # normalize the coordinates
        coords[...,0]=coords[...,0]/W
        coords[...,1]=coords[...,1]/H
        # # add the coordinates to the representation
        # representation[...,:2]=coords
        # representation.register_hook(lambda grad: print("representation",grad))
        # permute to have the non-matches axis first
        # KP is the non-matches axis, SF is the matches axis
        # # N X KP x SF x R
        # representation=representation.permute(0,2,1,3)
        # permute the coordinates to have the non-matches axis first
        # KP is the non-matches axis, SF is the matches axis
        # N X KP x SF x 2
        coords=coords.permute(0,2,1,3)
        # pass the representations to the MCL loss
        loss=self.mcl_loss(coords)
        # log the loss to wandb
        wandb.log({'patch_contrastive_loss':loss.item()})
        torch.cuda.empty_cache()
        return loss