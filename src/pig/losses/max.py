from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.nn.functional as F

from pig.utils.distance_matrix_visualizer import DistanceMatrixVisualizer

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MaxMatchLoss(nn.Module):
# A class to compute the contrastive loss over two axis (matches axis and non-matches axis)
# the input is of shape (batch_size, matches_axis, non_matches_axis, entity_size)
# where entities could be a vector or an image
# matches_axis is the same entitiy in different occurrences (over time or augmented)
# non_matches_axis is different entities (e.g. other keypoints)
    def __init__(self, config, prefix=''):
        super().__init__()
        self.sigma=config['sigma_for_mcl_soft']
        self.matches_loss_weight=config['matches_loss_weight']
        self.prefix=prefix
        self.count=0
        self.visualize_matrices=DistanceMatrixVisualizer()

    def forward(self, x):
        # N batch_size, M matches_axis, NM non_matches_axis, E entity_size
        N,NM,M,E=x.shape
        # compute the matches distance matrix
        # d( (N x NM x 1 x M x E) , (N x NM x M  x 1 x E)) = (N x NM x M x M)
        matches_dist_matrix=torch.norm(x[:,:,None,:,:]-x[:,:,:,None,:],dim=-1)
        # print(f"matches_dist_matrix={matches_dist_matrix[0,0]}")
        # # soft distance matrix
        matches_dist_matrix=torch.exp(-matches_dist_matrix/(2*self.sigma**2))
        # print(f"matches_dist_matrix={matches_dist_matrix[0,0]}")
        # input()
        # we want to minimize the maximum distance between matches
        # i.e. maximiaze the minimum soft distance
        # we want the max matches value to be 1
        # N x NM
        matches_loss= 1 - torch.amin(matches_dist_matrix,dim=(-1,-2))
        # print(f"matches_loss={matches_loss[0]}")
        # the mean of the loss
        matches_loss=torch.max(matches_loss)
        # print(f"matches_loss={matches_loss}")
        # compute the non-matches distance matrix
        # reshape to have all entities in the same axis
        # (N x NM x M x E) -> (N x NM*M x E)
        x=x.contiguous().view(N,NM*M,E)
        # compute the non-matches distance matrix
        # d( (N x 1 x NM*M x E) , (N x NM*M x 1 x E)) = (N x NM*M x NM*M)
        non_matches_dist_matrix=torch.norm(x[:,None,:,:]-x[:,:,None,:],dim=-1)
        # soft distance matrix
        non_matches_dist_matrix=torch.exp(-non_matches_dist_matrix/(2*self.sigma**2))
        # log the losses (with prefix)
        wandb.log({f'{self.prefix} Matches Loss':matches_loss.item()})
        # log the matches and non-matches distance matrices
        if self.count%100==0:
            self.visualize_matrices.log_matrices(matches_dist_matrix,non_matches_dist_matrix)
        self.count+=1
        # the loss to minimize is the contrastive loss and the matches loss
        loss =  self.matches_loss_weight * matches_loss
        # return the loss        
        return loss