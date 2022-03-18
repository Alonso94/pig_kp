from matplotlib.pyplot import step
import torch
import torch.nn as nn
import torch.nn.functional as F

from pig.utils.distance_matrix_visualizer import DistanceMatrixVisualizer

import wandb

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MatrixContrastiveLoss(nn.Module):
# A class to compute the contrastive loss over two axis (matches axis and non-matches axis)
# the input is of shape (batch_size, matches_axis, non_matches_axis, entity_size)
# where entities could be a vector or an image
# matches_axis is the same entitiy in different occurrences (over time or augmented)
# non_matches_axis is different entities (e.g. other keypoints)
    def __init__(self, config, prefix=''):
        super().__init__()
        self.margin=config['margin_for_matrix_contrastive_loss']
        self.sigma=config['sigma_for_mcl_soft']
        self.contrastive_loss_weight=config['contrastive_loss_weight']
        self.matches_loss_weight=config['matches_loss_weight']
        self.non_matches_loss_weight=config['non_matches_loss_weight']
        self.prefix=prefix
        self.count=0
        self.visualize_matrices=DistanceMatrixVisualizer()

    def forward(self, x):
        # N batch_size, M matches_axis, NM non_matches_axis, E entity_size
        N,NM,M,E=x.shape
        # compute the matches distance matrix
        # d( (N x NM x 1 x M x E) , (N x NM x M  x 1 x E)) = (N x NM x M x M)
        matches_dist_matrix=torch.norm(x[:,:,None,:,:]-x[:,:,:,None,:],dim=-1)
        # # soft distance matrix
        matches_dist_matrix=torch.exp(-matches_dist_matrix/(2*self.sigma**2))
        # sum for each batch
        # (N x NM x M x M) -> (N)
        matches_dist=matches_dist_matrix.sum(dim=(-1,-2,-3))
        # compute the non-matches distance matrix
        # reshape to have all entities in the same axis
        # (N x NM x M x E) -> (N x NM*M x E)
        x=x.contiguous().view(N,NM*M,E)
        # compute the non-matches distance matrix
        # d( (N x 1 x NM*M x E) , (N x NM*M x 1 x E)) = (N x NM*M x NM*M)
        non_matches_dist_matrix=torch.norm(x[:,None,:,:]-x[:,:,None,:],dim=-1)
        # soft distance matrix
        non_matches_dist_matrix=torch.exp(-non_matches_dist_matrix/(2*self.sigma**2))
        # sum for each batch
        # (N x NM*M x NM*M) -> (N)
        non_matches_dist=non_matches_dist_matrix.sum(dim=(-1,-2))
        # subtract the matches distance from the non-matches distance
        # (N)
        non_matches_dist=non_matches_dist-matches_dist
        # normalize the matches distance by the size of the matches arrays
        normalized_matches_dist=matches_dist/(NM*M*M)
        # normalize the non-matches distance by the size of the non-matches arrays - the size of the matches arrays
        normalized_non_matches_dist=non_matches_dist/(NM*M*(NM*M-M))
        # compute the contrastive loss
        contrastive_loss=torch.clamp(normalized_non_matches_dist-normalized_matches_dist+self.margin,min=0)
        # compute the matches loss (mean of the normalized matches distance)
        matches_loss=normalized_matches_dist.mean()
        # compute the non-matches loss (mean of the normalized non-matches distance)
        non_matches_loss=normalized_non_matches_dist.mean()
        # mean of the contrastive loss
        contrastive_loss=contrastive_loss.mean()
        # log the losses (with prefix)
        wandb.log({f'{self.prefix}/Contrastive Loss':contrastive_loss.item(),\
                    f'{self.prefix}/Matches Loss':matches_loss.item(),\
                    f'{self.prefix}/Non-Matches Loss':non_matches_loss.item()})
        # log the matches and non-matches distance matrices
        if self.count%100==0:
            self.visualize_matrices.log_matrices(matches_dist_matrix,non_matches_dist_matrix)
        self.count+=1
        # the loss to minimize is the contrastive loss and the matches loss
        loss = self.contrastive_loss_weight * contrastive_loss \
                + self.matches_loss_weight * (1-matches_loss) \
                + self.non_matches_loss_weight * non_matches_loss
        # return the loss        
        return loss