import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatialConsistencyLoss(nn.Module):
# The loss ensures the spatial consistency of the keypoints
# based on the principle of conservation of energy
# the keypoint has two types of energy:
# 1. kinetic energy: the sum of one half the square of its relative velocity to other keypoints
# 2. potential energy: the sum of the distances to the other keypoints

    def __init__(self, config):
        super(SpatialConsistencyLoss, self).__init__()
        self.width = config['width']
        self.height = config['height']
    
    def forward(self, coords):
        N,SF,KP,_ = coords.shape
        # coords.register_hook(lambda grad: print("coords",grad))
        # normalize the coordinates by the image size
        coords = coords / torch.tensor([self.width, self.height], device=device)
        # distance vectors between the keypoints in the same frame
        # N x SF x KP x KP x 2
        distances = coords[:,:,:,None,:]-coords[:,:,None,:,:]
        # velocity vectors for the keypoints over time
        # N x SF x SF x KP x KP x 2
        velocities = distances[:,:,None,:,:,:]-distances[:,None,:,:,:,:]
        # take just the first row of the velocity vectors
        # N x SF x KP x KP x 1
        velocities = velocities[:,0]
        # Potential energy
        # the potential energy at each time step is the sum of the distances
        # N x SF x KP x 2
        potential_energy = distances.sum(dim=-2)
        # Kinetic energy
        # the kinetic energy at each time step is the sum of half the square of the relative velocity
        # N x SF x KP x 2
        kinetic_energy = ((velocities**2)/2).sum(dim=-2)
        # the total energy at each time step is the sum of the kinetic and potential energy
        # N x SF x KP x 2
        total_energy = kinetic_energy + potential_energy
        # the difference between the total energy at each time step should be zero, sum for the coordinates
        # N x SF x KP
        total_energy_diff = abs(total_energy[:,:,:,None]-total_energy[:,:,None,:]).sum(dim=-1)
        # The loss is the mean of the total energy difference
        loss=total_energy_diff.mean()
        # log the loss to wandb
        wandb.log({'Spatial Consistency Loss': loss.item()})
        torch.cuda.empty_cache()
        return loss