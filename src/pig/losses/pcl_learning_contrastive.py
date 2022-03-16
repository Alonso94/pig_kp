from builtins import print
from pig.models.representation_model import RepresentationModel
from pig.losses.mcl import MatrixContrastiveLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.augmentation import RandomAffine

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500


import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchContrastiveLoss(nn.Module):
    # A representation agent that takes coordinates and return a representation of the patch around them
    # that can be used to compute the loss
    # it trains the RepresentationModel to learn the representation using the MCL loss
    def __init__(self, config):
        super().__init__()
        # initialize the representation model
        self.representation_model=RepresentationModel(config).to(device)
        # watch the representation model
        wandb.watch(self.representation_model)
        # MCL loss
        self.mcl_loss=MatrixContrastiveLoss(config)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.representation_model.parameters(), 
                                            lr=config['lr_for_representation_model'], 
                                            weight_decay=config['weight_decay_for_representation_model'])
        self.epochs=config['num_epochs_for_representation_model']
        # Random affine augmentation
        self.affine_transform=RandomAffine(p=0.9, degrees=(-30, 30), translate=(0.001, 0.001), scale=(0.8, 1.2)).to(device)
        self.matches_per_patch=config['matches_per_patch']
        self.representation_size=config['representation_size']
        self.noised_coords=config['noised_coords']
        # self.number_of_samples=config['number_of_samples']
        self.num_keypoints=config['num_keypoints']

    def threshold(self, fm):
        fm-=0.4
        fm=F.sigmoid(10000*fm)
        # visualize the thresholded gaussian
        # plt.imshow(fm[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        return fm

    def forward(self, coords, feature_maps, images):
        # if self.number_of_samples is not None:
        #     # draw samples
        #     samples=torch.randint(0,self.num_keypoints,(self.number_of_samples,),device=device)
        #     # get the samples
        #     coords=coords[:,:,samples,:]
        # extract patches
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # repeate the image for each keypoint
        # N x SF x KP x C x H x W
        images=images.unsqueeze(2).repeat(1,1,KP,1,1,1)
        # generate the mask
        # N x SF x KP x 1 x H x W
        mask=self.threshold(feature_maps).unsqueeze(3)
        # mask.register_hook(lambda grad: print("mask",grad)) # print gradients
        # extract the patches
        # N x SF x KP x C x H x W
        patches=images*mask
        # visualize a patch over time
        # for i in range(SF):
        #     plt.imshow(patches[0,i,0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        #     plt.show()
        # patches.register_hook(lambda grad: print("patches_1",grad)) # print gradients
        # pass patches through the representation model
        # N x SF x KP x R
        representations=self.representation_model(patches)
        # representations.register_hook(lambda grad: print("representations",grad)) # print gradients 
        # normalize the coordinates
        coords[...,0]=coords[...,0]/W
        coords[...,1]=coords[...,1]/H
        coords=coords.view(N,SF,KP,2)
        # add the coords to the representations
        # N x SF x KP x R+2
        representations=torch.cat((coords,representations),dim=-1)
        # permute to have the non-matches axis first
        # KP is the non-matches axis, SF is the matches axis
        # N X KP x SF x R
        representations=representations.permute(0,2,1,3)
        # compute the loss
        loss=self.mcl_loss(representations)
        # log the loss
        wandb.log({'patch_contrastive_loss':loss.item()})
        torch.cuda.empty_cache()
        return loss
    
    def train_representation(self, coords, feature_maps, images):
        # if self.number_of_samples is not None:
        #     # draw samples
        #     samples=torch.randint(0,self.num_keypoints,(self.number_of_samples,),device=device)
        #     # get the samples
        #     coords=coords[:,:,samples,:]
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # plt.imshow(images[0,0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # We use the matrix contrastive loss to train the representation model over the generated representations
        # the matches axis are the same patch augmented by random affine transformations
        # the non-matches are other patches
        # we will use the coordinates of the first frame
        # N x KP x 2
        coords=coords[:,0,:,:]
        if self.noised_coords:
            # add random number betweeb width and height to the coordinates
            coords[:,:,0]+=torch.randint(-int(W/8),int(W/8),(N,KP,)).to(device)
            coords[:,:,1]+=torch.randint(-int(H/8),int(H/8),(N,KP,)).to(device)
        # extract patches
        # repeate the first frame of the sequence of images for each keypoint
        # N x 1 x KP x C x H x W
        images=images[:,0,:,:,:].unsqueeze(1).repeat(1,KP,1,1,1).unsqueeze(1)
        # generate the mask
        # N x SF x KP x 1 x H x W
        mask=self.threshold(feature_maps).unsqueeze(3)
        # extract the patches
        # N x SF x KP x C x H x W
        patches=images*mask
        # plt.imshow(patches[0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # plt.imshow(patches[1,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # plt.show()
        # permute and repeate the patches by the matches_per_batch
        # N x M x KP x C x H x W
        patches=patches.repeat(1,self.matches_per_patch,1,1,1,1)
        # reshape the patches
        # N*M*KP x C x H x W
        patches=patches.view(-1,C,H,W)
        # stop the gradient for the patches
        patches=patches.detach()
        # apply the random affine transform
        # N*M*KP x C x H x W
        augmented_patches=self.affine_transform(patches)
        # reshape the augmented patches
        # N x M x KP x C x H x W
        augmented_patches=augmented_patches.view(N,SF*self.matches_per_patch,KP,C,H,W)
        # visualize the augmented patches
        # for i in range(self.matches_per_patch):
        #     plt.imshow(augmented_patches[0,i,0,:,:,:].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        #     plt.show()
        # iterate by the number of epochs
        for _ in range(self.epochs):#, leave=False):
            self.optimizer.zero_grad()
            # pass patches through the representation model
            # N x M x KP x R
            representations=self.representation_model(augmented_patches)
            # permute to have the non-matches axis first
            # non-matches axis is KP, macthes axis is M
            # N x KP x M x R
            representations=representations.permute(0,2,1,3)
            # compute the loss
            loss=self.mcl_loss(representations)
            # backprop
            loss.backward()
            self.optimizer.step()
            del loss
        torch.cuda.empty_cache()

