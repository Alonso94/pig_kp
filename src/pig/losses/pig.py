from pig.entropy_layer.entropy import Entropy
from pig.joint_entropy.joint_entropy import JointEntropy
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
        self.joint_entropy=JointEntropy(config['region_size'],config['bandwidth']).to(device)
        self.masked_entropy_loss_weight=config['masked_entropy_loss_weight']
        self.overlapping_loss_weight=config['overlapping_loss_weight']
        self.num_keypoints=config['num_keypoints']
        # extract patches
        self.patch_extractor=PatchExtractor(config, std=config['std_for_featuremap_generation'], aggregate=True)
        self.count=0

    def log_masked_image(self, image, masked_image):
        # plot the image and the entropy
        # create a subplot
        fig,axes=plt.subplots(1,2)
        # plot the original image
        axes[0].imshow(image[:3].detach().cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1])
        axes[0].set_title('Original image')
        # plot the entropy of the RGB channels
        axes[1].imshow(masked_image.detach().cpu().numpy(),cmap='jet')
        axes[1].set_title('Masked entropy')
        plt.tight_layout()
        # plt.show()
        # log the image to wandb
        wandb.log({'masked_image':wandb.Image(fig)})
        plt.close('all')

    def threshold(self, fm):
        # # plot the fm as 3d surface
        # fig=plt.figure()
        # ax=fig.add_subplot(111, projection='3d')
        # X = np.arange(0,fm.shape[3],1)
        # Y = np.arange(0,fm.shape[2],1)
        # X, Y = np.meshgrid(X, Y)
        # ax.plot_surface(X,Y,fm[0,0].detach().cpu().numpy(),cmap='jet')
        # plt.show()
        fm-=0.5
        fm=F.sigmoid(10000*fm)
        # # visualize the thresholded gaussian
        # plt.imshow(fm[0,0].detach().cpu().numpy(),cmap='gray')
        # save figure locally
        # plt.savefig('thresholded_gaussian.png')
        return fm

    def forward(self, feature_maps, status, images):
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # feature_maps.register_hook(lambda grad: print("coords_pig",grad.mean()))
        # grad mean = 2.3e-11
        # depth_entropy=self.entropy_layer(images[:,:,-1].unsqueeze(2))[:,:,0]
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        # plt.imshow(rgb_entropy[0,0].detach().cpu().numpy(),cmap='jet')
        # save figure locally
        # plt.savefig('rgb_entropy.png')
        # plot_entropy(images[0,0],rgb_entropy[0])
        # joint_entropy=self.joint_entropy(images)
        # conditional_entropy=joint_entropy-depth_entropy
        # plot_joint_entropy(images[0],joint_entropy[0,0],'Conditional')
        # sum the entropy of each image
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        # penalize the background by subtract -0.1
        rgb_entropy-=0.1
        # generate the gaussians around keypoints
        # aggregated_mask=self.patch_extractor(coords, size=(H, W)).to(device)
        # multiply the feature maps with the status
        feature_maps= status[...,None,None]*feature_maps
        aggregated_feature_maps=feature_maps.sum(dim=2)
        aggregated_mask=self.threshold(aggregated_feature_maps)
        # print(aggregated_mask.shape)
        # grad mean =  2.7e-7
        # masked depth entropy
        masked_depth_entropy=rgb_entropy*aggregated_mask
        # plt.imshow(masked_depth_entropy[0,0].detach().cpu().numpy(),cmap='jet')
        # save figure locally
        # plt.savefig('masked_depth_entropy.png')
        # input()
        # print("masked_depth_entropy",masked_depth_entropy.shape)
        # normalize the masked depth entropy
        # masked_depth_entropy=(min_depth_entropy-masked_depth_entropy)/(min_depth_entropy-max_depth_entropy)
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        masked_entropy_sum=torch.sum(masked_depth_entropy,dim=(-1,-2))
        # masked_entropy_sum.register_hook(lambda grad: print("masked_entropy_sum",grad.mean()))
        # grad mean =  5.1e-7
        masked_entropy_loss=1-masked_entropy_sum/rgb_entropy_sum
        # penalize the overlapping of the patches
        # the maximum of the aggregated feature maps should as small as possible
        overlapping_loss = aggregated_feature_maps.amax(dim=(-1,-2))
        # masked_entropy_loss=(min_depth_entropy-masked_entropy_sum)/(min_depth_entropy-max_depth_entropy)
        # print("masked_entropy_loss",masked_entropy_loss[0,0])
        # masked_entropy_loss.register_hook(lambda grad: print("masked_entropy_loss",grad.mean()))
        # grad mean =  0.0104
        # the pig loss
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss + self.overlapping_loss_weight*overlapping_loss
        # mean over time
        pig_loss=pig_loss.mean(dim=-1)
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # grad mean =  1.0
        if self.count%100==0:
            self.log_masked_image(images[0,0],masked_depth_entropy[0,0])
        self.count+=1
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item(),
                   'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
                   'pig/overlapping_loss':overlapping_loss.mean().item()})
        torch.cuda.empty_cache()
        return pig_loss

