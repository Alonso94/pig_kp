from cmath import nan
from pig.entropy_layer.entropy import Entropy
from pig.utils.plot_entropy_histogram import *
from pig.utils.extract_patches import PatchExtractor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchInfoGainLoss(nn.Module):
# Loss function which computes the information gain of the patches
# we compute the entropy and th conditional of the patches around the keypoints
# the aim to maximize the percentage of entropy of the depth channels of an image
# and also to maximize the percentage of conditional entropy of the whole image
    def __init__(self, config):
        super().__init__()
        self.entropy_layer=Entropy(config['region_size'],config['bandwidth']).to(device)
        self.masked_entropy_loss_weight=config['masked_entropy_loss_weight']
        self.active_overlapping_weight=config['active_overlapping_weight']
        self.inactive_overlapping_weight=config['inactive_overlapping_weight']
        self.num_keypoints=config['num_keypoints']
        self.status_weight=config['status_weight']
        self.fm_threshold=config['fm_threshold']
        self.thresholded_fm_scale=config['thresholded_fm_scale']
        self.schedule=config['schedule']
        self.patches_weight=config['patches_weight']
        self.acceleration_weight=config['acceleration_weight']
        self.count=0

    def forward(self, coords, feature_maps, active_status, dynamic_status, images):
        N,SF,KP,_=coords.shape
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # distance travelled by all keypoints in each time frame
        # N x SF x KP
        shifted_coordinates=torch.roll(coords,-1,dims=1)
        distance_travelled=torch.norm(shifted_coordinates-coords,dim=-1)
        # velocity of keypoints
        shifted_distances=torch.roll(distance_travelled,-1,dims=1)
        velocity=torch.abs(shifted_distances-distance_travelled)
        # acceleration of keypoints
        shifted_velocities=torch.roll(velocity,-1,dims=1)
        acceleration=torch.abs(shifted_velocities-velocity)
        acceleration=torch.sum(acceleration,dim=-1)
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        # N x SF x KP
        # sum the entropy of each image
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        # threshold the feature maps
        # N x SF x KP x H x W
        feature_maps=feature_maps-self.fm_threshold
        feature_maps=self.thresholded_fm_scale*F.relu(feature_maps)
        # the patch around each keypoint is the feature map multiplied by the rgb entropy
        # N x SF x KP x H x W
        patches=feature_maps*rgb_entropy[:,:,None,:,:]
        shifted_patches=torch.roll(patches,-1,dims=1)
        # the patches loss is the difference between the patches and the shifted patches
        # N x SF x KP x H x W
        patches_loss=torch.abs(patches-shifted_patches)
        # mask inactive keypoints
        patches_loss=patches_loss*active_status[:,:,:,None,None]
        # sum for each keypoint
        # N x SF x KP
        patches_loss_sum=patches_loss.sum(dim=(-1,-2))
        # normalize the patches loss
        # N x SF x KP
        patches_loss_sum=patches_loss_sum/1000
        # sum for each time frame
        # N x SF
        patches_loss_sum=patches_loss_sum.sum(dim=-1)
        # static and dynamic feature maps
        active_feature_maps=active_status[...,None,None]*feature_maps
        # status_feature_maps.register_hook(lambda grad: print("fm*status_pig",grad.mean()))
        # generate the gaussians around keypoints
        # N x SF x H x W
        aggregated_active_feature_maps=active_feature_maps.sum(dim=2)
        aggregated_active__mask=torch.clamp(aggregated_active_feature_maps,min=0,max=1)
        # masked entropy
        masked_entropy=rgb_entropy*aggregated_active__mask
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        masked_entropy_sum=torch.sum(masked_entropy,dim=(-1,-2))
        masked_entropy_loss=1-masked_entropy_sum/(rgb_entropy_sum+1e-10)
        # penalize the overlapping of the patches
        # the maximum of the aggregated feature maps should as small as possible
        active_overlapping_loss = active_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        inactive_overlapping_loss=(feature_maps-active_feature_maps).sum(dim=2).amax(dim=(-1,-2))
        # the pig loss
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss\
                    + self.active_overlapping_weight*active_overlapping_loss \
                    + self.inactive_overlapping_weight*inactive_overlapping_loss \
                    + self.status_weight*active_status.sum(dim=-1).mean() \
                    + self.acceleration_weight*acceleration.mean()
        if masked_entropy_loss.mean()<self.schedule:
            self.patches_weight*patches_loss_sum.mean()
        # mean over time
        # N
        pig_loss=pig_loss.mean(dim=-1)
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item(),
            'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
            'pig/active_overlapping_loss':active_overlapping_loss.mean().item(),
            'pig/inactive_overlapping_loss':inactive_overlapping_loss.mean().item(),
            'pig/acceleration':acceleration.mean().item(),
            'pig/active_status_loss':active_status.sum(dim=2).mean().item(),
            'pig/patches_loss':patches_loss_sum.mean().item(),
            })
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

