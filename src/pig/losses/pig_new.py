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
        self.conditional_entropy_loss_weight=config['conditional_entropy_loss_weight']
        self.active_overlapping_weight=config['active_overlapping_weight']
        self.inactive_overlapping_weight=config['inactive_overlapping_weight']
        self.dynamic_overlapping_weight=config['dynamic_overlapping_weight']
        self.static_movement_loss_weight=config['static_movement_loss_weight']
        self.dynamic_movement_loss_weight=config['dynamic_movement_loss_weight']
        self.num_keypoints=config['num_keypoints']
        self.status_weight=config['status_weight']
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
        fm=fm-thresh
        fm=F.sigmoid(10000*fm)
        # visualize the thresholded gaussian
        # plt.imshow(fm[0,0,0].detach().cpu().numpy(),cmap='gray')
        # plt.show()
        return fm

    def forward(self, coords, feature_maps, active_status, images):
        N,SF,KP,_=coords.shape
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # # normalize the coordintaes
        # normalize_corrdintaes=coords/torch.tensor([H,W]).float().to(device)
        # distance travelled by all keypoints in each time frame
        # N x SF x KP
        shifted_coordinates=torch.roll(coords,1,dims=1)
        distance_travelled=torch.norm(shifted_coordinates-coords,dim=-1)
        distance_travelled[:,0,:]=0
        # velocity of keypoints
        shifted_distances=torch.roll(distance_travelled,1,dims=1)
        velocity=torch.abs(shifted_distances-distance_travelled)
        velocity[:,:2,:]=0
        # movement of all keypoint in each frame
        # N x SF x KP
        if SF>3:
            # acceleration of keypoints
            shifted_velocities=torch.roll(velocity,1,dims=1)
            acceleration=torch.abs(shifted_velocities-velocity)
            acceleration[:,:3,:]=0
            movement=acceleration
        else:
            movement=velocity
        # sum for each keypoint
        # N x KP
        movement=torch.sum(movement,dim=1)
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        shifted_rgb_entropy=torch.roll(rgb_entropy,1,dims=1)
        joint_entropy=torch.maximum(rgb_entropy,shifted_rgb_entropy)
        conditional_entropy=joint_entropy-rgb_entropy
        conditional_entropy-=0.8
        conditional_entropy=F.relu(conditional_entropy)
        # sum the entropy of each image
        # N x SF
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        conditional_entropy_sum=conditional_entropy.sum(dim=(-1,-2))
        # multiply the feature maps with the status
        # N x SF x KP x H x W
        active_feature_maps=active_status[...,None,None]*feature_maps
        # find the masked conditional entropy for each keypoint
        # N x SF x KP x H x W
        masked_fm_conditional_entropy=active_feature_maps*conditional_entropy[:,:,None,:,:]
        # movement loss
        # maximum allowed velocity is the maximum of the masked conditional entropy
        # N x SF x KP
        max_conditional_entropy=torch.amax(masked_fm_conditional_entropy,dim=(-1,-2))
        # movement status
        # get 1 if max conditional entropy is higher than 1 (there is movement)
        # and 0 otherwise (static)
        # N x SF x KP
        movement_status=torch.sigmoid(1000*(max_conditional_entropy-1))
        # multiply the drop indices for each keypoint
        # N x KP
        general_movement_status=torch.prod(movement_status,dim=1)
        general_active_status=torch.prod(active_status,dim=1)
        # the dynamic movement 
        # drop the keypoint if it is not moving
        dynamic_movement= movement * general_movement_status
        # the static movement
        # drop the keypoint if it is not active or if it is moving
        static_movement = ( movement - dynamic_movement ) * general_active_status
        # mean over the keypoints
        # N 
        static_movement_loss=torch.max(static_movement,dim=-1)
        dynamic_movement_loss=torch.mean(dynamic_movement,dim=-1)
        # aggregate feature maps of all keypoints
        # N x SF x H x W
        aggregated_active_feature_maps=active_feature_maps.sum(dim=2)
        # aggregated_active_mask=self.threshold(aggregated_active_feature_maps, self.fm_threshold)
        aggregated_active_mask=torch.clamp(aggregated_active_feature_maps,min=0,max=1)
        # mask the entropy
        # N x SF x H x W
        masked_entropy=rgb_entropy*aggregated_active_mask
        # if self.count%100==0:
        #     self.log_masked_image(images[0,0],masked_entropy[0,0])
        masked_conditional_entropy= conditional_entropy*aggregated_active_mask
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        # sum the masked entropy for each time frame
        # N x SF
        masked_entropy_sum=torch.sum(masked_entropy,dim=(-1,-2))
        # sum the masked conditional entropy for each time frame
        # N x SF
        masked_conditional_entropy_sum=torch.sum(masked_conditional_entropy,dim=(-1,-2))
        # the loss is the percebtage of the entropy in the masked regions
        masked_entropy_loss=1-masked_entropy_sum/(rgb_entropy_sum+1e-10)
        masked_conditional_entropy_loss=1-masked_conditional_entropy_sum/(conditional_entropy_sum+1e-10)
        # penalize the overlapping of the patches
        # we drop the dynamic feature maps
        # N x SF x KP x H x W
        drop_moving_feature_maps = feature_maps * (1-movement_status)[:,:,:,None,None]
        dynamic_feature_maps = feature_maps - drop_moving_feature_maps
        # the maximum of the aggregated feature maps should as small as possible
        # N x SF
        overlapping_loss = drop_moving_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        inactive_feature_maps = feature_maps * (1-active_status)[:,:,:,None,None]
        inactive_feature_maps_sum = inactive_feature_maps.sum(dim=2)
        inactive_overlapping_loss = inactive_feature_maps_sum.amax(dim=(-1,-2))
        # # plot the fm as 3d surface
        # # surface=drop_moving_feature_maps.sum(dim=2)[0,0]
        # surface=feature_maps[0,0,0]
        # fig=plt.figure()
        # ax=fig.add_subplot(111, projection='3d')
        # X = np.arange(0,surface.shape[1],1)
        # Y = np.arange(0,surface.shape[0],1)
        # X, Y = np.meshgrid(X, Y)
        # ax.plot_surface(X,Y,surface.detach().cpu().numpy(),cmap='jet')
        # plt.show()
        # penalize the dynamic feature maps
        # N x SF
        dynamic_overlapping_loss= dynamic_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        # status loss is the sum of active status
        # N x SF
        status_loss=active_status.sum(dim=-1)
        # the pig loss
        # N x SF
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss\
                    + self.conditional_entropy_loss_weight*masked_conditional_entropy_loss\
                    + self.active_overlapping_weight*overlapping_loss \
                    + self.inactive_overlapping_weight*inactive_overlapping_loss \
                    + self.dynamic_overlapping_weight*dynamic_overlapping_loss \
                    + self.status_weight*status_loss  
        # if masked_entropy_loss.max()<self.schedule:
        #     pig_loss+= self.activation_loss_weight * activation_loss \
        #             + self.status_weight*status_loss
        # mean over time
        # N
        pig_loss=pig_loss.mean(dim=-1)
        # if masked_entropy_loss.max()<self.schedule:
        pig_loss+= self.static_movement_loss_weight * static_movement_loss \
            + self.dynamic_movement_loss_weight * dynamic_movement_loss
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item(),
            'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
            'pig/conditional_entropy_percentage':masked_conditional_entropy_loss.mean().item(),
            'pig/overlapping_loss':overlapping_loss.mean().item(),
            'pig/inactive_overlapping_loss':inactive_overlapping_loss.mean().item(),
            'pig/dynamic_overlapping_loss':dynamic_overlapping_loss.mean().item(),
            'pig/active_status_loss':status_loss.mean().item(),
            'pig/static_movement_loss':static_movement_loss.mean().item(),
            'pig/dynamic_movement_loss':dynamic_movement_loss.mean().item(),
            })
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

