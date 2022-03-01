from pig.entropy_layer.entropy import Entropy
from pig.joint_entropy.joint_entropy import JointEntropy
from pig.histogram_layer.histogram import Histogram

from pig.utils.plot_entropy_histogram import *
from pig.utils.extract_patches import PatchExtractor

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['font.size']= 5

import torch
import torch.nn as nn

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
        self.pig_loss_weight=config['pig_loss_weight']

        # extract patches
        self.patch_extractor=PatchExtractor(config, std=config['std_for_featuremap_generation'], aggregate=True)

    def plot_masked_image(self, image, masked_image):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(1,2)
        # plot the original image
        axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1].imshow(masked_image.detach().cpu().numpy(),cmap='jet')
        axes[1].set_title('Masked entropy')
        plt.show()

    def forward(self, coords, images):
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # print(coords.shape)
        depth_entropy=self.entropy_layer(images[:,:,-1].unsqueeze(2))[:,:,0]
        # plot_entropy(images[0,0],depth_entropy[0,0].unsqueeze(0))
        # joint_entropy=self.joint_entropy_layer(images)
        # self.plot_joint_entropy(images[0,:2],joint_entropy[0,1],label='Joint')
        # conditiona_netropy=joint_entropy-entropy
        # self.plot_joint_entropy(images[0,:2],conditiona_netropy[0,1],label='Conditional')
        # coords.register_hook(lambda grad: print(grad))
        # generate the gaussians around keypoints
        aggregated_mask=self.patch_extractor(coords, size=(H, W)).to(device)
        # masked depth entropy
        masked_depth_entropy=depth_entropy*aggregated_mask
        # self.plot_masked_image(images[0,0],masked_depth_entropy[0,0])
        # the pig loss
        pig_loss = 1 - torch.sum(masked_depth_entropy)/torch.sum(depth_entropy)
        pig_loss = self.pig_loss_weight*pig_loss
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item()})
        # print('pig_loss:',pig_loss.item())
        torch.cuda.empty_cache()
        return pig_loss

