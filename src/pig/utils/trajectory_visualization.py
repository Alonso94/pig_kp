import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 300
import matplotlib._color_data as mcd
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import os
import random
import wandb

import time

class TrajectoryVisualizer():
    def __init__(self):
        keys=random.sample(list(mcd.XKCD_COLORS.keys()), wandb.config['num_keypoints'])
        # self.colors=[mcd.XKCD_COLORS[k] for k in keys]
        self.counter=0

    def log_video(self,images,coords, active_status, dynamic_status, label=None):
        start=time.time()
        # images=images.detach().cpu().permute(0,2,3,1).numpy()
        coords=coords.detach().cpu().numpy().astype(np.int32)[0]
        active_status=active_status.detach().cpu().numpy().astype(np.int32)[0]
        dynamic_status=dynamic_status.detach().cpu().numpy().astype(np.int32)[0]
        fig,ax=plt.subplots(1)
        lines = []
        # green color if status is 1 else red
        colors=['g' if s==1 else 'y' for s in active_status[0]]
        for i in range(len(colors)):
            lobj = ax.plot([coords[0,i,0]],[coords[0,i,1]],lw=1,color=colors[i])[0]
            lines.append(lobj)
        scat = ax.scatter([coords[0,:,0]],[coords[0,:,1]], s=30, c=colors, marker='o')
        trajectory_length=np.zeros(len(colors), dtype=np.int32)
        def update(i):
            ax.imshow(images[i][:,:,::-1])
            for lnum,line in enumerate(lines):
                colors[lnum]='g' if active_status[i,lnum]==1 else 'y'
                if dynamic_status[i,lnum]==1 : colors[lnum]='b' 
                line.set_color(colors[lnum])
                if active_status[i,lnum]>0.1:
                    # set data for each line separately.
                    num_points=max(i-min(trajectory_length[lnum],5),0)
                    line.set_data(coords[num_points:i,lnum,0], coords[num_points:i,lnum,1])
                    trajectory_length[lnum]+=1
                    # update the coords array for the disappearing points
                else:
                    trajectory_length[lnum]=0
                    # coords[i,lnum,0]=-1
                    # coords[i,lnum,1]=-1
            scat.set_color(colors)
            scat.set_offsets(coords[i,:,:])
            return lines, scat
        animation=FuncAnimation(fig, update, frames=len(images), interval=20, repeat=False)
        writer=PillowWriter(fps=20)
        animation.save('animation_{0}_{1}.gif'.format(label,self.counter), writer=writer)
        video=wandb.Video('animation_{0}_{1}.gif'.format(label,self.counter),format="gif")
        # print('logging video finished in {0} seconds'.format(time.time()-start))
        wandb.log({'kp_trajectories':video})
        os.remove('animation_{0}_{1}.gif'.format(label,self.counter))
        plt.close('all')