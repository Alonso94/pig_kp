import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PatchExtractor(nn.Module):
# a calss to extract patches around keypoints from a given image
    def __init__(self, config, mask=True):
        super().__init__()
        # threshold for featuremap generation
        # self.threshold=torch.nn.Threshold(-1*config['threshold_for_featuremaps'],1)
        # std for featuremap generation
        self.std_for_featuremap_generation=config['std_for_featuremap_generation']
        self.fm_threshold=config['fm_threshold']
        self.thresholded_fm_scale=config['thresholded_fm_scale']
    
    def forward(self, coords, size):
        # coords.register_hook(lambda grad: print("coords extract patches",grad.mean()))
        # grad mean = 2.3e-11
        N,SF,KP,_=coords.shape
        H,W=size
        std=torch.tensor(self.std_for_featuremap_generation).to(device)
        pi=torch.tensor(3.14159265358979323846).to(device)
        # pixel coordinates
        # x - N*KP x H x W x 1
        x = torch.arange(0,W,device=device).view(1,1,-1).repeat(N*SF*KP,H,1)
        # y - N*KP x H x W x 1
        y = torch.arange(0,H,device=device).view(1,-1,1).repeat(N*SF*KP,1,W)
        # pixels - N x H x W x 2
        pixels=torch.stack([x,y],dim=-1)
        # reshape the coordinates
        # N*KP x 2
        coords=coords.contiguous().view(N*SF*KP,2)
        # coords.register_hook(lambda grad: print("coords repeat",grad.mean()))
        # grad mean = 2.3e-11
        # adjust the coordinates to a compatible shape
        coords=coords.view(N*SF*KP,1,1,2).repeat(1,H,W,1)
        # coords.register_hook(lambda grad: print("coords input gaussian",grad.mean()))
        # grad mean = 6e-16
        # calculate the gaussian
        # squared_distances - N*KP x H x W x 1
        squared_distances=torch.sum((pixels-coords)**2,dim=-1)
        # exp_term - N*KP x H x W
        exp_term=torch.exp(-1*(squared_distances)/(2*std**2))
        # gaussian - N*KP x H x W 
        # remove the normalization term to make penalizing the overlapping easier
        # fm=exp_term#/(2*std**2*torch.sqrt(2*pi))
        # reshape the fm
        # N x KP x H x W
        fm=exp_term.view(N,SF,KP,H,W)
        # threshold the featuremap
        fm=fm-self.fm_threshold
        fm=self.thresholded_fm_scale*F.relu(fm)
        fm=torch.clamp(fm,min=0,max=1)
        # # visualize the process
        # fig,axes=plt.subplots(1,4, constrained_layout=True, figsize=(20,7))
        # fig.tight_layout()
        # fig.suptitle('Patch extraction for keypoint \n with coordinatees {}'.format(coords[0,0,0].detach().cpu().numpy()), fontsize=35)
        # axes[0].imshow(squared_distances[0].detach().cpu().numpy(), cmap='jet')
        # axes[0].set_title('Squared Distances', fontsize=25)
        # axes[1].imshow(exp_term[0].detach().cpu().numpy(), cmap='jet')
        # axes[1].set_title('Gaussian', fontsize=25)
        # axes[3].imshow(fm[0,0,0].detach().cpu().numpy(), cmap='jet')
        # axes[3].set_title('Thresholded \n Gaussian', fontsize=25)
        # # remove the ticks
        # for ax in axes.flat:
        #     ax.axis("off")
        # plt.savefig('patch_extraction.png',dpi=1000, transparent=True)
        # # kill all the figures
        # plt.close('all')
        return fm
