import torch

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300

import wandb

class DistanceMatrixVisualizer():
    def __init__(self, prefix=''):
        self.prefix=prefix

    def log_matrices(self, matches_dist, non_matches_dist):
        min_v=torch.min(torch.min(matches_dist),torch.min(non_matches_dist))
        max_v=torch.max(torch.max(matches_dist),torch.max(non_matches_dist))
        fig, ax =plt.subplots()
        im=ax.matshow(matches_dist[0,0].detach().cpu().numpy(), cmap='jet', vmin = min_v, vmax = max_v)
        ax.set_title('Matches distance matrix' '\n''(same entity in different occurrences)')
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.85)
        plt.colorbar(im,fraction=0.05, pad=0.05)
        plt.tight_layout()
        # log the figure with the prefix
        wandb.log({'Matches Distance Matrix':wandb.Image(fig)})
        # plt.show()
        plt.close(fig)
        fig, ax =plt.subplots()
        im=ax.matshow(non_matches_dist[0].detach().cpu().numpy(), cmap='jet', vmin = min_v, vmax = max_v)
        ax.set_title('Distance matrix' '\n''(Comparing all entities)')
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.85)
        plt.colorbar(im,fraction=0.05, pad=0.05)
        plt.tight_layout()
        # log the figure with the prefix
        wandb.log({'Non-Matches Distance Matrix':wandb.Image(fig)})
        # plt.show()
        plt.close(fig)

