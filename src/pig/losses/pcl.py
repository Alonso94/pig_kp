from turtle import forward
from pig.histogram_layer.histogram import Histogram
from pig.losses.mcl import MatrixContrastiveLoss
from pig.utils.extract_patches import PatchExtractor
from pig.utils.plot_entropy_histogram import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import sobel

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchContrastiveLoss(nn.Module):
# A class to compute the patch contrastive loss between keypoints
# the class takes a sequence of images and coordinates of keypoints
# extract the patches around the keypoints and find representation of each patch
# uing the Matrix Contrastive loss on the representations of the patches
# we get the patch contrastive loss value
    def __init__(self, config):
        super().__init__()
        # patch extractor
        self.patch_extractor=PatchExtractor(config, std=config['std_for_featuremap_generation'], aggregate=False)
        # histogram layer
        self.histogram_layer=Histogram(config['bandwidth']).to(device)
        # MCL loss
        self.mcl_loss=MatrixContrastiveLoss(config)
        # number of samples
        self.num_samples=config['num_samples']
        # number of keypoints
        self.num_keypoints=config['num_keypoints']

    def threshold(self, fm):
        fm-=0.001
        fm=F.sigmoid(10000*fm)
        # visualize the thresholded gaussian
        # plt.imshow(fm[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        return fm

    def forward(self, feature_maps, images):
        if self.num_samples is not None:
            # draw samples
            samples=torch.randint(0,self.num_keypoints,(self.num_samples,),device=device)
            # get the samples
            feature_maps=feature_maps[:,:,samples,:,:]
        # # get rid of the depth channel
        # images=images[:,:,:3]
        # feature_maps.register_hook(lambda grad: print("coords_pcl",grad.mean()))
        # extract patches
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # reshape the images
        images=images.view(N*SF,C,H,W)
        # apply sobel filter
        images=sobel(images)
        # reshape the images back
        images=images.view(N,SF,C,H,W)
        # repeate the image for each keypoint
        # N x SF x KP x C x H x W
        images=images.unsqueeze(2).repeat(1,1,KP,1,1,1)
        # generate the mask
        # N x SF x KP x 1 x H x W
        mask=self.threshold(feature_maps).unsqueeze(3)
        # mask.register_hook(lambda grad: print("mask",grad.mean()))
        # the sum of the masks for normalization
        # N x SF x KP
        mask_sum=mask.sum(dim=(-1,-2)).mean()
        # get the patches
        # N x SF x KP x C x H x W
        patches=images*mask
        # reshape the patches
        # N*SF*KP x C x H x W
        patches=patches.view(N*SF*KP,C,H,W)
        # plt.imshow(patches[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # apply sobel filter to get the edges
        # N*SF*KP x C x H x W
        # patches=sobel(patches)
        # plt.imshow(patches[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # get the histogram of the patches
        # N*SF*KP x C x 256
        # patches.register_hook(lambda grad: print("patches",grad.mean()))
        hist=self.histogram_layer(patches)
        # normalize the histogram
        # N*SF*KP x C x 256
        hist=hist/mask_sum
        # get rid of the black pixels
        hist[:,:,0]=0
        # print(hist.shape)
        # plot_histogram(patches[0],hist[0])
        # reshape the histogram
        # N x SF x KP x C*256
        hist=hist.view(N,SF,KP,C*256)
        # permute to have the SF as a non-matches axis and the KP as a matches axis
        # N x KP x SF x C*256
        hist=hist.permute(0,2,1,3)
        # pass the histogram to the MCL loss
        loss=self.mcl_loss(hist)
        # log the loss
        wandb.log({'patch_contrastive_loss':loss.item()})
        torch.cuda.empty_cache()
        return loss


        
        