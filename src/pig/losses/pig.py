from pig.entropy_layer.entropy import Entropy

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
        self.pig_loss_weight=config['pig_loss_weight']
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

    def forward(self, coords, images):
        N,SF,KP,_=coords.shape
        N,SF,C,H,W=images.shape
        # coords.register_hook(lambda grad: print("coords_pig",grad.mean()))
        # grad mean = 2.3e-11
        depth_entropy=self.entropy_layer(images[:,:,-1].unsqueeze(2))[:,:,0]
        # sum the entropy of each image
        depth_entropy_sum=depth_entropy.sum(dim=(-1,-2))
        # generate the gaussians around keypoints
        aggregated_mask=self.patch_extractor(coords, size=(H, W)).to(device)
        # print(aggregated_mask.shape)
        # grad mean =  2.7e-7
        # masked depth entropy
        masked_depth_entropy=depth_entropy*aggregated_mask
        # we want to encourage maximizing the entropy in the masked regions
        # at the same time encourage our keypoints to spread out
        masked_entropy_sum=torch.sum(masked_depth_entropy,dim=(-1,-2))
        # masked_entropy_sum.register_hook(lambda grad: print("masked_entropy_sum",grad.mean()))
        # grad mean =  5.1e-7
        masked_entropy_loss=1-masked_entropy_sum/depth_entropy_sum
        # masked_entropy_loss.register_hook(lambda grad: print("masked_entropy_loss",grad.mean()))
        # grad mean =  0.0104
        # the pig loss
        pig_loss = self.pig_loss_weight*(masked_entropy_loss)
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
                   'pig/masked_entropy_percentage':masked_entropy_loss.mean().item()})
        torch.cuda.empty_cache()
        return pig_loss

