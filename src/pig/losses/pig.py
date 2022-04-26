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
        self.conditional_entropy_loss_weight=config['conditional_entropy_loss_weight']
        self.active_overlapping_weight=config['active_overlapping_weight']
        self.inactive_overlapping_weight=config['inactive_overlapping_weight']
        self.dynamic_overlapping_weight=config['dynamic_overlapping_weight']
        self.static_overlapping_weight=config['static_overlapping_weight']
        # self.static_loss_weight=config['static_loss_weight']
        # self.dynamic_loss_weight=config['dynamic_loss_weight']
        self.static_movement_loss_weight=config['static_movement_loss_weight']
        self.dynamic_movement_loss_weight=config['dynamic_movement_loss_weight']
        self.margin=config['margin']
        self.contrastive_loss_weight=config['contrastive_loss_weight']
        self.num_keypoints=config['num_keypoints']
        self.status_weight=config['status_weight']
        self.dynamic_status_weight=config['dynamic_status_weight']
        self.fm_threshold=config['fm_threshold']
        self.thresholded_fm_scale=config['thresholded_fm_scale']
        self.schedule=config['schedule']
        self.penalize_background=config['penalize_background']
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
        plt.show()
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
        # fm=3*F.relu(fm)
        # # visualize the thresholded gaussian
        # plt.imshow(fm.detach().cpu().numpy(),cmap='gray')
        # plt.show()
        # save figure locally
        # plt.savefig('thresholded_gaussian.png')
        return fm

    def forward(self, coords, feature_maps, active_status, dynamic_status, images):
        N,SF,KP,_=coords.shape
        N,SF,KP,H,W=feature_maps.shape
        N,SF,C,H,W=images.shape
        # distance travelled by all keypoints in each time frame
        # N x SF x KP
        shifted_coordinates=torch.roll(coords,-1,dims=1)
        # print("coords",coords[0,1])
        # print("shifted_coordinates",shifted_coordinates[0,0])
        # shifted_coordinates.register_hook(lambda grad: print("shifted_coordinates_pig",grad.mean()))
        distance_travelled=torch.norm(shifted_coordinates-coords,dim=-1)
        # print("d",distance_travelled[0,0,:])
        # input()
        # mask the inactive keypoint by multiplying by status
        # distance_travelled=distance_travelled*
        # velocity of dynamic keypoints
        shifted_distances=torch.roll(distance_travelled,-1,dims=1)
        # print(shifted_distances[0,0,:])
        velocity=torch.abs(shifted_distances-distance_travelled)
        # print("v",velocity[0,0,:])
        # input()
        # acceleration of dynamic keypoints
        shifted_velocities=torch.roll(velocity,-1,dims=1)
        # print("shifted_velocities",shifted_velocities[0,0,:])
        acceleration=torch.abs(shifted_velocities-velocity)
        # print("a",acceleration[0,0,:])
        # input()
        # # mask static keypoints by multiplying by dynamic status
        # velocity=velocity*dynamic_status
        # # mask dynamic keypoints in distance travelled
        # distance_travelled=distance_travelled*active_status-distance_travelled*dynamic_status
        # sum for for each frame
        # N x SF
        # distance_travelled=torch.sum(distance_travelled,dim=-1)
        # velocity=torch.sum(velocity,dim=-1)
        # acceleration=torch.sum(acceleration,dim=-1)
        acceleration_static=(active_status-dynamic_status)*acceleration
        acceleration_dynamic=dynamic_status*acceleration
        acceleration_static=torch.sum(acceleration_static,dim=-1)
        acceleration_dynamic=torch.sum(acceleration_dynamic,dim=-1)
        # we want the distance travelled by the dynamic keypoints to be higher than the static keypoints
        # we use contrastive loss to achieve this
        # N x SF
        static_distance_travelled=torch.amax(distance_travelled*(active_status-dynamic_status),dim=-1)
        dynamic_distance_travelled=torch.amin(distance_travelled*dynamic_status,dim=-1)
        contrastive_loss=torch.clamp(static_distance_travelled-dynamic_distance_travelled+self.margin,min=0)
        # N x SF x H x W
        rgb_entropy=self.entropy_layer(images[:,:,:3])[:,:,0]
        shifted_rgb_entropy=torch.roll(rgb_entropy,1,dims=1)
        joint_entropy=torch.maximum(rgb_entropy,shifted_rgb_entropy)
        conditional_entropy=joint_entropy-rgb_entropy
        conditional_entropy-=0.8
        conditional_entropy=F.relu(conditional_entropy)
        # N x SF x KP
        # sum the entropy of each image
        rgb_entropy_sum=rgb_entropy.sum(dim=(-1,-2))
        conditional_entropy_sum=conditional_entropy.sum(dim=(-1,-2))
        # # penalize active keypoints in the background
        # if self.penalize_background:
        #     rgb_entropy-=0.005
        #     conditional_entropy-=0.005
        # plot_conditional_entropy(images[0],conditional_entropy[0,1],joint_entropy[0,1],rgb_entropy[0])
        # distance_travelled.register_hook(lambda grad: print("distance_travelled",grad.mean()))
        # status.register_hook(lambda grad: print("status",grad.mean()))
        # multiply the feature maps with the status
        # threshold the feature maps
        # N x SF x KP x H x W
        feature_maps=feature_maps-self.fm_threshold
        feature_maps=self.thresholded_fm_scale*F.relu(feature_maps)
        # static and dynamic feature maps
        active_feature_maps=active_status[...,None,None]*feature_maps
        dynamic_feature_maps=dynamic_status[...,None,None]*feature_maps
        # status_feature_maps.register_hook(lambda grad: print("fm*status_pig",grad.mean()))
        # generate the gaussians around keypoints
        # N x SF x H x W
        aggregated_active_feature_maps=active_feature_maps.sum(dim=2)
        aggregated_active__mask=torch.clamp(aggregated_active_feature_maps,min=0,max=1)
        aggregated_dynamic_featue_maps=dynamic_feature_maps.sum(dim=2)
        aggregated_dynamic_mask=torch.clamp(aggregated_dynamic_featue_maps,min=0,max=1)
        # fig=plt.figure()
        # ax=fig.add_subplot(111, projection='3d')
        # X = np.arange(0,aggregated_mask.shape[3],1)
        # Y = np.arange(0,aggregated_mask.shape[2],1)
        # X, Y = np.meshgrid(X, Y)
        # ax.plot_surface(X,Y,aggregated_mask[0,0].detach().cpu().numpy(),cmap='jet')
        # plt.show()
        # plt.imshow(aggregated_static__mask[0,0].detach().cpu().numpy(),cmap='jet')
        # plt.show()
        # plt.imshow(aggregated_dynamic_mask[0,0].detach().cpu().numpy(),cmap='jet')
        # plt.show()
        masked_entropy=rgb_entropy*aggregated_active__mask
        # self.log_masked_image(images[0,0],masked_entropy[0,0])
        masked_conditional_entropy=conditional_entropy*aggregated_dynamic_mask
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        masked_entropy_sum=torch.sum(masked_entropy,dim=(-1,-2))
        masked_conditional_entropy_sum=torch.sum(masked_conditional_entropy,dim=(-1,-2))
        # masked_entropy_sum.register_hook(lambda grad: print("masked_entropy_sum_pig",grad.mean()))
        masked_entropy_loss=1-masked_entropy_sum/(rgb_entropy_sum+1e-10)
        masked_conditional_entropy_loss=1-masked_conditional_entropy_sum/(conditional_entropy_sum+1e-10)
        # penalize the overlapping of the patches
        # the maximum of the aggregated feature maps should as small as possible
        # if masked_entropy_loss.mean()<self.schedule:
        #     overlapping_loss = status_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        # else:
        active_overlapping_loss = active_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        static_overlapping_loss = (active_feature_maps-dynamic_feature_maps).sum(dim=2).amax(dim=(-1,-2))
        dynamic_overlapping_loss = dynamic_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        inactive_overlapping_loss=(feature_maps-active_feature_maps).sum(dim=2).amax(dim=(-1,-2))
        # dynamic_overlapping_loss = dynamic_feature_maps.sum(dim=2).amax(dim=(-1,-2))
        # masked_entropy_loss=rgb_entropy_sum-masked_entropy_sum
        # masked_entropy_loss.register_hook(lambda grad: print("masked_entropy_loss",grad.mean()))
        # the pig loss
        pig_loss = self.masked_entropy_loss_weight*masked_entropy_loss\
                    + self.conditional_entropy_loss_weight*masked_conditional_entropy_loss\
                    + self.dynamic_status_weight*dynamic_status.sum(dim=-1).mean() \
                    + self.status_weight*(active_status-dynamic_status).sum(dim=-1).mean() \
                    + self.active_overlapping_weight*active_overlapping_loss \
                    + self.static_overlapping_weight*static_overlapping_loss \
                    + self.dynamic_overlapping_weight*dynamic_overlapping_loss \
                    + self.inactive_overlapping_weight*inactive_overlapping_loss  \
                    + self.static_movement_loss_weight*acceleration_static \
                    + self.contrastive_loss_weight*contrastive_loss \
                    + self.dynamic_movement_loss_weight*acceleration_dynamic 
                    # + self.status_weight*active_status.sum(dim=-1).mean()
        # if masked_entropy_loss.max()<self.schedule:
        #     pig_loss+= self.static_loss_weight*distance_travelled
        #     pig_loss+= self.status_weight*active_status.sum(dim=-1).mean()
        # if masked_conditional_entropy_loss.mean()<self.schedule:
        #     pig_loss+= self.contrastive_loss_weight*contrastive_loss \
        #             + self.dynamic_movement_loss_weight*acceleration_dynamic
        #     pig_loss+= self.status_weight*active_status.sum(dim=-1).mean()
        #     pig_loss+= self.dynamic_status_weight*dynamic_status.sum(dim=-1).mean()
        #     pig_loss+= self.dynamic_loss_weight*velocity
        # mean over time
        # N
        pig_loss=pig_loss.mean(dim=-1)
        # mean over the batch
        pig_loss=pig_loss.mean()
        # pig_loss.register_hook(lambda grad: print("pig_loss",grad.mean()))
        # log to wandb
        wandb.log({'pig_loss':pig_loss.item(),
            'pig/masked_entropy_percentage':masked_entropy_loss.mean().item(),
            'pig/conditional_entropy_percentage':masked_conditional_entropy_loss.mean().item(),
            'pig/active_overlapping_loss':active_overlapping_loss.mean().item(),
            'pig/inactive_overlapping_loss':inactive_overlapping_loss.mean().item(),
            'pig/static_overlapping_loss':static_overlapping_loss.mean().item(),
            'pig/dynamic_overlapping_loss':dynamic_overlapping_loss.mean().item(),
            'pig/contrastive_loss':contrastive_loss.mean().item(),
            # 'pig/distance_travelled':distance_travelled.mean().item(),
            # 'pig/velocity':velocity.mean().item(),
            'pig/acceleration_static': acceleration_static.mean().item(),
            'pig/acceleration_dynamic': acceleration_dynamic.mean().item(),
            'pig/active_status_loss':active_status.sum(dim=2).mean().item(),
            'pig/dynamic_status_loss':dynamic_status.sum(dim=2).mean().item(),
            })
        self.count+=1
        torch.cuda.empty_cache()
        return pig_loss

