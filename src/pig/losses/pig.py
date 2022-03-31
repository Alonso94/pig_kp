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
        self.overlapping_loss_weight=config['overlapping_loss_weight']
        self.movement_loss_weight=config['movement_loss_weight']
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

    def threshold(self, fm, thresh):
        # # plot the fm as 3d surface
        # fig=plt.figure()
        # ax=fig.add_subplot(111, projection='3d')
        # X = np.arange(0,fm.shape[3],1)
        # Y = np.arange(0,fm.shape[2],1)
        # X, Y = np.meshgrid(X, Y)
        # ax.plot_surface(X,Y,fm[0,0].detach().cpu().numpy(),cmap='jet')
        # plt.show()
        fm=fm-thresh
        fm=F.sigmoid(10000*fm)
        # # visualize the thresholded gaussian
        # plt.imshow(fm[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        # save figure locally
        # plt.savefig('thresholded_gaussian.png')
        return fm

    def forward(self, coords, feature_maps, status, images):
        N,SF,KP,_=coords.shape
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # feature_maps.register_hook(lambda grad: print("coords_pig",grad.mean()))
        # grad mean = 2.3e-11
        # depth_entropy=self.entropy_layer(images[:,:,-1].unsqueeze(2))[:,:,0]
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        # plt.imshow(rgb_entropy[0,0].detach().cpu().numpy(),cmap='jet')
        # save figure locally
        # plt.savefig('rgb_entropy.png')
        # plot_entropy(images[0,0],rgb_entropy[0])
        # N x SF x H x W
        shifted_rgb_entropy=torch.roll(rgb_entropy,1,dims=1)
        # N x SF x H x W
        joint_entropy=torch.maximum(rgb_entropy,shifted_rgb_entropy)
        # N x SF x H x W
        conditional_entropy=joint_entropy-rgb_entropy
        # plot_conditional_entropy(images[0],conditional_entropy[0,1],joint_entropy[0,1], rgb_entropy[0])
        # repeat the conditional entropy for each keypoint
        # N x SF x KP x H x W
        conditional_entropy=conditional_entropy.unsqueeze(2).repeat(1,1,KP,1,1)
        # mask for conditional entropy
        mask=self.threshold(feature_maps.clone(),0.95)
        # multiply conditional entropy by feature maps
        # N x SF x KP x H x W
        masked_conditional_entropy=conditional_entropy*mask
        # sum conditional entropy for each keypoint
        # N x SF x KP
        masked_conditional_entropy=masked_conditional_entropy.sum(dim=(-1,-2))
        # normalize the masked conditional entropy by the size of the mask
        # N x SF x KP
        masked_conditional_entropy=masked_conditional_entropy/mask.sum(dim=(-1,-2))
        # sum over the time
        # N x KP
        masked_conditional_entropy=masked_conditional_entropy.sum(dim=1)
        # distance travelled by each keypoint
        # N x KP
        distance_travelled=torch.norm(coords[:,1:]-coords[:,:-1],dim=-1).sum(dim=1)
        # The movemement loss ia loss to discourage the movement of the keypoints if the conditional entropy is low
        # N x KP
        movement_loss=torch.abs(distance_travelled-10*masked_conditional_entropy)
        # print('movement loss',movement_loss.shape)
        movement_loss=movement_loss.mean(dim=-1)
        # sum the entropy of each image
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        # penalize the background by subtract -0.1
        # rgb_entropy-=0.1
        # multiply the feature maps with the status
        # print('status featuremap sum',status,(status[...,None,None]*feature_maps).sum(dim=(-1,-2)))
        # print("status sum",status[0,0,0].sum(dim=(-1,-2)).shape)
        feature_maps= status[...,None,None]*feature_maps
        # generate the gaussians around keypoints
        # aggregated_mask=self.patch_extractor(coords, size=(H, W)).to(device)
        aggregated_feature_maps=feature_maps.sum(dim=2)
        # penalize the overlapping of the patches
        # the maximum of the aggregated feature maps should as small as possible
        overlapping_loss = aggregated_feature_maps.amax(dim=(-1,-2))
        # print(aggregated_mask.shape)
        # grad mean =  2.7e-7
        # masked depth entropy
        aggregated_mask=self.threshold(aggregated_feature_maps,0.25)
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
        # masked_entropy_loss=(min_depth_entropy-masked_entropy_sum)/(min_depth_entropy-max_depth_entropy)
        # print("masked_entropy_loss",masked_entropy_loss[0,0])
        # masked_entropy_loss.register_hook(lambda grad: print("masked_entropy_loss",grad.mean()))
        # grad mean =  0.0104
        # the pig loss
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss + self.overlapping_loss_weight*overlapping_loss
        # mean over time
        pig_loss=pig_loss.mean(dim=-1)
        pig_loss+= self.movement_loss_weight*movement_loss
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # grad mean =  1.0
        # if self.count%100==0:
        #     self.log_masked_image(images[0,0],masked_depth_entropy[0,0])
        # log to wandb
        if self.count%10==0:
            wandb.log({'pig_loss':pig_loss.item(),
                   'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
                   'pig/overlapping_loss':overlapping_loss.mean().item(),
                   'pig/movement_loss':movement_loss.mean().item(),})
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

