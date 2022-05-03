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
        self.dynamic_overlapping_weight=config['dynamic_overlapping_weight']
        self.mode_switching_weight=config['mode_switching_weight']
        self.activation_loss_weight=config['activation_loss_weight']
        self.static_movement_loss_weight=config['static_movement_loss_weight']
        self.dynamic_movement_loss_weight=config['dynamic_movement_loss_weight']
        self.num_keypoints=config['num_keypoints']
        self.status_weight=config['status_weight']
        self.schedule=config['schedule']
        self.penalize_background=config['penalize_background']
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

    def forward(self, coords, feature_maps, active_status, dynamic_status, images):
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
        # # consider only active keypoints
        # movement = movement * active_status
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
        # # activate keypoints according to the entropy sum
        # # N x SF
        # shifted_rgb_entropy_sum=torch.roll(rgb_entropy_sum,1,dims=1)
        # entropy_diff=(rgb_entropy_sum-shifted_rgb_entropy_sum)/500
        # shifted_status=torch.roll(active_status,1,dims=1)
        # status_diff=active_status.sum(dim=-1)-shifted_status.sum(dim=-1)
        # activation_loss=torch.abs(entropy_diff-status_diff)
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
        # get 0 if max conditional entropy is higher than 1 (there is movement)
        # and 1 otherwise (static)
        # N x SF x KP
        movement_status=torch.sigmoid(1000*(1-max_conditional_entropy))
        # multiply the drop indices for each keypoint
        # N x KP
        general_movement_status=torch.prod(movement_status,dim=1)
        active_movement=movement*torch.prod(active_status,dim=1)
        # multiply th movement staus with the movement
        # to drop the dynamic keypoints
        # N x KP
        static_movement = movement * general_movement_status
        dynamic_movement=active_movement-static_movement
        # mean over the keypoints
        # N 
        static_movement_loss=torch.mean(static_movement,dim=-1)
        dynamic_movement_loss=torch.mean(dynamic_movement,dim=-1)
        # # discourage mode switching
        # # we want to minimize the points that track the movement
        # # this happens by penalizing the switch of the mode
        # # N x SF x KP
        # shifted_movement_status=torch.roll(movement_status,1,dims=1)
        # # xor the movement status
        # # N x SF x KP
        # mode_switch_status=torch.abs(movement_status-shifted_movement_status)
        # # sum the movement status over time, then clamp the sum to 1
        # # N x KP
        # # mode_switching_loss= torch.clamp(torch.sum(1-movement_status, dim=1),max=1)
        # mode_switching_loss=torch.sum(mode_switch_status,dim=1)
        # # mean over all the keypoints
        # # N
        # mode_switching_loss=torch.sum(mode_switching_loss,dim=-1)
        # aggregate feature maps of all keypoints
        # N x SF x H x W
        aggregated_active_feature_maps=active_feature_maps.sum(dim=2)
        # aggregated_active_mask=self.threshold(aggregated_active_feature_maps, self.fm_threshold)
        aggregated_active_mask=torch.clamp(aggregated_active_feature_maps,min=0,max=1)
        # mask the entropy
        # N x SF x H x W
        masked_entropy=rgb_entropy*aggregated_active_mask
        if self.count%500==0:
            self.log_masked_image(images[0,0],masked_entropy[0,0])
        # # visualize the process
        # fig,axes=plt.subplots(1,3, constrained_layout=True, figsize=(20,7))
        # fig.tight_layout()
        # plt.subplots_adjust(wspace=0.2)
        # fig.suptitle('Masked Entropy', fontsize=5)
        # axes[0].imshow(rgb_entropy[0,0].detach().cpu().numpy(), cmap='jet')
        # axes[0].set_title('RGB Entropy', fontsize=5)
        # axes[1].imshow(aggregated_active_mask[0,0].detach().cpu().numpy(), cmap='gray')
        # axes[1].set_title('Aggregated mask', fontsize=5)
        # axes[2].imshow(masked_entropy[0,0].detach().cpu().numpy(), cmap='jet')
        # axes[2].set_title('Masked entropy', fontsize=5)
        # # remove the ticks
        # for ax in axes.flat:
        #     ax.axis("off")
        # plt.show()
        # # kill all the figures
        # plt.close('all')
        # input('Press Enter to continue...')
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
        drop_moving_feature_maps = feature_maps * movement_status[:,:,:,None,None]
        dynamic_feature_maps = feature_maps - drop_moving_feature_maps
        # the maximum of the aggregated feature maps should as small as possible
        # N x SF
        overlapping_loss = drop_moving_feature_maps.sum(dim=2).amax(dim=(-1,-2))
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
            'pig/dynamic_overlapping_loss':dynamic_overlapping_loss.mean().item(),
            'pig/active_status_loss':status_loss.mean().item(),
            'pig/static_movement_loss':static_movement_loss.mean().item(),
            'pig/dynamic_movement_loss':dynamic_movement_loss.mean().item(),
            })
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

