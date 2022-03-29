from pig.losses.mcl import MatrixContrastiveLoss
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
        # dilation kernel for featuremap generation
        self.dilation_kernel=torch.ones(config['dilation_kernel_size'],config['dilation_kernel_size'],device=device)
        # number of samples
        self.number_of_samples=config['num_samples']
        self.iterations=config['number_of_rings_around_the_keypoint']
        self.num_keypoints=config['num_keypoints']

    def threshold(self, fm):
        fm-=0.7
        fm=F.sigmoid(10000*fm)
        # visualize the thresholded gaussian
        # plt.imshow(fm[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        return fm
    
    def forward(self, coords, feature_maps, status, images):
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
        # repeate the image for each keypoint
        # N x SF x KP x C x H x W
        images=images.unsqueeze(2).repeat(1,1,KP,1,1,1)
        # generate the mask
        # N x SF x KP x 1 x H x W
        mask=self.threshold(feature_maps).unsqueeze(3)
        # reshape the mask
        # N*SF*KP x 1 x H x W
        mask=mask.view(N*SF*KP,1,H,W)
        # reshape the images
        # N*SF*KP x C x H x W
        images=images.view(N*SF*KP,C,H,W)
        # initialize the representation vector
        # N*SF*KP x 2 (coordinates) + n_iterations + 3 (edges)
        representation=torch.zeros((N*SF*KP,5+self.iterations),device=device)
        # highest possible value of the sum of pixel values is used for normalization
        highest_possible_sum=(3*255*mask.sum(dim=(1,2,3)))
        # old values are using to compute the ring around the keypoint
        old_highest_possible_sum=torch.zeros_like(highest_possible_sum).to(device)
        old_patch_sum=torch.zeros_like(highest_possible_sum).to(device)
        # iterate to extract the sum of pixel values in a ring around the keypoint
        for i in range(self.iterations):
            # extract the patch
            # N*SF*KP x C x H x W
            patch=images*mask
            # plt.imshow(patch[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
            # plt.show()
            # sum the values
            patch_sum=patch.sum(dim=(1,2,3))
            # subtract the old values to get the ring around the previous mask
            representation[:,i+2]=(patch_sum-old_patch_sum)
            old_patch_sum=patch_sum
            # normalization
            representation[:,i+2]/=(highest_possible_sum-old_highest_possible_sum)
            mask=dilation(mask,self.dilation_kernel)
            old_highest_possible_sum=highest_possible_sum
            highest_possible_sum=(3*255*mask.sum(dim=(1,2,3)))
        # extract the patches (mask is a circle around the keypoint)
        # N*SF*KP x C x H x W
        patches=images*mask
        # plt.imshow(patches[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # use the sobel filter to extract the edges
        # N*SF*KP x 3 x H x W
        edges=sobel(patches)
        # plt.imshow(edges[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # edges.register_hook(lambda grad: print("representation",grad))
        # add the edges to the representation
        representation[:,2+self.iterations:]=edges.sum(dim=(-1,-2)).squeeze()
        # normalization
        representation[:,2+self.iterations:]/=(100*mask.sum(dim=(-1,-2)))
        # reshape the representations
        # N x SF x KP x R
        representation=representation.view(N,SF,KP,-1)
        # normalize the coordinates
        coords[...,0]=coords[...,0]/W
        coords[...,1]=coords[...,1]/H
        # add the coordinates to the representation
        representation[...,:2]=coords
        # representation.register_hook(lambda grad: print("representation",grad))
        # permute to have the non-matches axis first
        # KP is the non-matches axis, SF is the matches axis
        # N X KP x SF x R
        representation=representation.permute(0,2,1,3)
        status=status.permute(0,2,1)
        # pass the representations to the MCL loss
        loss=self.mcl_loss(representation, status)
        # log the loss to wandb
        wandb.log({'patch_contrastive_loss':loss.item()})
        torch.cuda.empty_cache()
        return loss