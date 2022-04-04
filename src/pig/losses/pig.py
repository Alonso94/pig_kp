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
        self.status_weight=config['status_weight']
        self.fm_threshold=config['fm_threshold']
        self.schedule=config['schedule']
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
        # plt.imshow(fm.detach().cpu().numpy(),cmap='gray')
        # plt.show()
        # save figure locally
        # plt.savefig('thresholded_gaussian.png')
        return fm

    def forward(self, coords, feature_maps, status, images):
        N,SF,KP,_=coords.shape
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # distance travelled by all keypoints in each time frame
        # N x SF
        shifted_coordinates=torch.roll(coords,1,dims=1)
        # shifted_coordinates.register_hook(lambda grad: print("shifted_coordinates_pig",grad.mean()))
        distance_travelled=torch.norm(shifted_coordinates-coords,dim=-1).sum(dim=-1)
        # coords.register_hook(lambda grad: print("coords grad", grad.mean()))
        # feature_maps.register_hook(lambda grad: print("fm_pig",grad.mean()))
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        # N x SF x KP
        # sum the entropy of each image
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        # penalize keypoints in the background
        rgb_entropy-=0.1
        # distance_travelled.register_hook(lambda grad: print("distance_travelled",grad.mean()))
        # status.register_hook(lambda grad: print("status",grad.mean()))
        # multiply the feature maps with the status
        status_feature_maps= status[...,None,None]*feature_maps
        # feature_maps.register_hook(lambda grad: print("fm*status_pig",grad.mean()))
        # generate the gaussians around keypoints
        aggregated_feature_maps=status_feature_maps.sum(dim=2)
        # aggregated_feature_maps.register_hook(lambda grad: print("aggregated_feature_maps_pig",grad.mean()))
        # overlapping_loss.register_hook(lambda grad: print("overlapping_loss",grad.mean()))
        # masked depth entropy
        aggregated_mask=self.threshold(aggregated_feature_maps,self.fm_threshold)
        # aggregated_mask.register_hook(lambda grad: print("aggregated_mask",grad.mean()))
        masked_depth_entropy=rgb_entropy*aggregated_mask
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        masked_entropy_sum=torch.sum(masked_depth_entropy,dim=(-1,-2))
        # masked_entropy_sum.register_hook(lambda grad: print("masked_entropy_sum_pig",grad.mean()))
        masked_entropy_loss=1-masked_entropy_sum/rgb_entropy_sum
        # penalize the overlapping of the patches
        # the maximum of the aggregated feature maps should as small as possible
        if masked_entropy_loss.mean()<self.schedule:
            overlapping_loss = aggregated_feature_maps.amax(dim=(-1,-2))
        else:
            overlapping_loss = feature_maps.sum(dim=2).amax(dim=(-1,-2))
        # masked_entropy_loss=rgb_entropy_sum-masked_entropy_sum
        # masked_entropy_loss.register_hook(lambda grad: print("masked_entropy_loss",grad.mean()))
        # the pig loss
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss \
                    + self.overlapping_loss_weight*overlapping_loss \
                    + self.movement_loss_weight*distance_travelled
        if masked_entropy_loss.mean()<self.schedule:
            pig_loss+= self.status_weight*status.sum(dim=-1).mean()
        # mean over time
        pig_loss=pig_loss.mean(dim=-1)
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item(),
                'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
                'pig/overlapping_loss':overlapping_loss.mean().item(),
                'pig/movement_loss':distance_travelled.mean().item(),
                'pig/status_loss':status.sum(dim=2).mean().item()})
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

