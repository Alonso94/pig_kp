from pig.entropy_layer.entropy import Entropy
from pig.joint_entropy.joint_entropy import JointEntropy
from pig.histogram_layer.histogram import Histogram

import torch
import torch.nn as nn

import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5

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
        self.histogram_layer=Histogram(config['bandwidth']).to(device)
        self.joint_entropy_layer=JointEntropy(config['region_size'],config['bandwidth']).to(device)

    def plot_entropy(self,image,entropy):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(1,3)
        # plot the original image
        axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
        axes[1].set_title('Entropy of the RGB channels')
        # plot the entropy of the depth channels
        axes[2].imshow(entropy[0].detach().cpu().numpy(),cmap='jet')
        axes[2].set_title('Entropy of the depth channels')
        # remove the axis ticks and the borders
        for ax in axes.flat:
            # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.axis('off')
        plt.show()
    
    def plot_joint_entropy(self,image,joint_entropy, label=None):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(2,2)
        # plot the original image
        axes[0,0].imshow(image[0,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0,1].imshow(image[1,:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        # axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1,0].imshow(joint_entropy[0].detach().cpu().numpy(),cmap='jet')
        axes[1,0].set_title('{0} entropy of the RGB channels'.format(label))
        # plot the entropy of the depth channels
        axes[1,1].imshow(joint_entropy[1].detach().cpu().numpy(),cmap='jet')
        axes[1,1].set_title('{0} entropy of the depth channels'.format(label))
        # remove the axis ticks and the borders
        for ax in axes.flat:
            # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
            ax.axis('off')
        plt.show()

    def plot_histogram(self,image,histogram):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(1,2)
        # plot the original image
        axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0].axis('off')
        # axes[0].set_title('Original image')
        # plot the histogram of the image
        axes[1].bar(range(256),histogram[0].detach().cpu().numpy(),color='r',alpha=0.3)
        axes[1].bar(range(256),histogram[1].detach().cpu().numpy(),color='b',alpha=0.3)
        axes[1].bar(range(256),histogram[2].detach().cpu().numpy(),color='g',alpha=0.3)
        axes[1].set_title('Histogram of the RGB channels')
        # remove the axis ticks and the borders
        # for ax in axes.flat:
        #     # ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        #     ax.axis('off')
        plt.show()

    def forward(self, coords, images):
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # hist=self.histogram_layer(images)
        # self.plot_histogram(images[0,0],hist[0,0])
        # compute depth entropy
        depth_entropy=self.entropy_layer(images[:,:,-1].unsqueeze(2))
        self.plot_entropy(images[0,0],depth_entropy[0,0])
        # joint_entropy=self.joint_entropy_layer(images)
        # self.plot_joint_entropy(images[0,:2],joint_entropy[0,1],label='Joint')
        # conditiona_netropy=joint_entropy-entropy
        # self.plot_joint_entropy(images[0,:2],conditiona_netropy[0,1],label='Conditional')