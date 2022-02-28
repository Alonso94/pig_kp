from pig.entropy_layer.entropy import Entropy
# from pig.joint_entropy.joint_entropy import JointEntropy

import torch
import torch.nn as nn

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500

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
        # self.joint_entropy_layer=JointEntropy(config['region_size'],config['bandwidth']).to(device)

    def plot_entropy(self,image,entropy):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(3,1)
        # plot the original image
        axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
        # axes[1].set_title('Entropy of the RGB channels')
        # plot the entropy of the depth channels
        axes[2].imshow(entropy[1].detach().cpu().numpy(),cmap='jet')
        # axes[2].set_title('Entropy of the depth channels')
        # remove the axis ticks and the borders
        for ax in axes.flat:
            # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.axis('off')
        plt.show()
    
    def plot_joint_entropy(self,image,joint_entropy):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(2,2)
        # plot the original image
        axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1,0].imshow(joint_entropy[0].detach().cpu().numpy(),cmap='jet')
        # axes[1].set_title('Entropy of the RGB channels')
        # plot the entropy of the depth channels
        axes[1,1].imshow(joint_entropy[1].detach().cpu().numpy(),cmap='jet')
        # axes[2].set_title('Entropy of the depth channels')
        # remove the axis ticks and the borders
        for ax in axes.flat:
            # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.axis('off')
        plt.show()

    def forward(self, coords, images):
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # compute the entropy of the RGB channels   
        entropy=self.entropy_layer(images)
        self.plot_entropy(images[0,0],entropy[0,0])
        # joint_entropy=self.joint_entropy_layer(images)
        # self.plot_joint_entropy(images[0,:2],joint_entropy[0,1])
        # conditiona_netropy=joint_entropy-entropy
        # self.plot_joint_entropy(images[0,:2],conditiona_netropy[0,1])