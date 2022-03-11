import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib._color_data as mcd
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import os
import random
import wandb

class TrajectoryVisualizer():
    def __init__(self):
        keys=random.sample(list(mcd.XKCD_COLORS.keys()), wandb.config['num_keypoints'])
        self.colors=[mcd.XKCD_COLORS[k] for k in keys]
        self.counter=0

    def log_video(self,images,coords, label=None):
        # images=images.detach().cpu().permute(0,2,3,1).numpy()
        coords=coords.detach().cpu().numpy().astype(np.int32)[0]
        fig,ax=plt.subplots(1)
        lines = []
        for i in range(len(self.colors)):
            lobj = ax.plot([coords[0,i,0]],[coords[0,i,1]],lw=1,color=self.colors[i])[0]
            lines.append(lobj)
        scat = ax.scatter([coords[0,:,0]],[coords[0,:,1]], s=30, c=self.colors, marker='o')
        trajectry_length=5
        def update(i):
            ax.imshow(images[i][:,:,::-1])
            for lnum,line in enumerate(lines):
                # set data for each line separately.
                line.set_data(coords[max(i-trajectry_length,0):i,lnum,0], coords[max(i-trajectry_length,0):i,lnum,1])
            scat.set_offsets(coords[i,:,:])
            return lines, scat
        animation=FuncAnimation(fig, update, frames=len(images), interval=20, repeat=False)
        writer=PillowWriter(fps=20)
        animation.save('animation_{0}_{1}.gif'.format(label,self.counter), writer=writer)
        video=wandb.Video('animation_{0}_{1}.gif'.format(label,self.counter),format="gif")
        wandb.log({'kp_trajectories':video})
        # os.remove('animation_{0}_{1}.gif'.format(label,self.counter))
        plt.close('all')