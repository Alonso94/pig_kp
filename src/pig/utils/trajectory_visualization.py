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
    def __init__(self, config):
        keys=random.sample(list(mcd.XKCD_COLORS.keys()), config['num_keypoints'])
        # self.colors=[mcd.XKCD_COLORS[k] for k in keys]
        # create a folder to store the videos with the same name as the model name
        self.folder_path='videos/'+config['model_name']
        os.makedirs(self.folder_path, exist_ok=True)
        self.counter=0

    def log_video(self,images,coords, active_status, label=None):
        print('Logging video : {0}'.format(label))
        start=time.time()
        # images=images.detach().cpu().permute(0,2,3,1).numpy()
        coords=coords.detach().cpu().numpy().astype(np.int32)[0]
        active_status=active_status.detach().cpu().numpy().astype(np.int32)[0]
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
                colors[lnum]='g' if active_status[i,lnum]>=0.95 else 'y'
                line.set_color(colors[lnum])
                if active_status[i,lnum]>0.1:
                    # set data for each line separately.
                    num_points=max(i-min(trajectory_length[lnum],5),0)
                    line.set_data(coords[num_points:i,lnum,0], coords[num_points:i,lnum,1])
                    trajectory_length[lnum]+=1
                    # update the coords array for the disappearing points
                else:
                    trajectory_length[lnum]=-1
            scat.set_color(colors)
            scat.set_offsets(coords[i,:,:])
            return lines, scat
        animation=FuncAnimation(fig, update, frames=len(images), interval=20, repeat=False)
        writer=PillowWriter(fps=5)
        animation.save('{0}/{1}.gif'.format(self.folder_path,label), writer=writer)
        # print('Drawing keypoints the video finished in {0} seconds'.format(time.time()-start))
        video=wandb.Video('{0}/{1}.gif'.format(self.folder_path,label),format="gif")
        wandb.log({'Videos/{0}'.format(label):video})
        plt.close('all')
        print('Logging the video took {0} seconds'.format(time.time()-start))