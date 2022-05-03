from pig.data.dataset_from_SIMITATE import DatasetFromSIMITATE
from pig.models.kpae import Encoder
from pig.utils.extract_patches import PatchExtractor
from pig.utils.trajectory_visualization import TrajectoryVisualizer
from pig.losses.pig_new import PatchInfoGainLoss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import wandb
import numpy as np
from tqdm import tqdm, trange

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PIG_agent(nn.Module):
    def __init__(self,config):
        super().__init__()
        # load the dataset
        self.dataset=DatasetFromSIMITATE(config)
        # self.dataset.show_sample(530)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # initialize the model
        self.model=Encoder(config).to(device)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        # initialize the pig loss
        self.pig_loss=PatchInfoGainLoss(config)
        # initialize the patch extractor
        self.patch_extractor=PatchExtractor(config).to(device)
        # initialize the wandb
        wandb.watch(self.model,log_freq=1000)
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer()
        self.log_video=config['log_video']
        self.log_video_every=config['log_video_every']
        self.save=config['save_model']
        self.epochs=config['epochs']
        self.palindrome=config['palindrome']
        self.palindrome_loss=nn.MSELoss()
        # self.log_trajectory()
        # input()

    def log_trajectory(self):
        # get the data
        sample=self.dataset.sample_video_from_data(100)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            data=torch.tensor(sample).float().permute(0,3,1,2).to(device).unsqueeze(0)
            kp=self.model(data)
            coords=kp[...,:2]
            active_status=kp[...,2]
            dynamic_status=kp[...,3]
            self.visualizer.log_video(sample[...,:3],coords, active_status, dynamic_status)
        self.model.train()

    def train(self):
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            # for sample in self.dataloader:
            for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # Training the model using human data
                # permute the data and move them to the device, enable the gradients
                data=sample.float().permute(0,1,4,2,3).to(device)
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # get the output
                kp=self.model(data)
                coords=kp[...,:2]
                active_status=kp[...,2]
                dynamic_status=kp[...,3]
                # generate feature maps around keypoints
                feature_maps=self.patch_extractor(coords, data.shape[-2:])
                # compute the loss
                loss=0
                loss+=self.pig_loss(coords.clone(), feature_maps.clone(), active_status, dynamic_status, data)
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                # update the parameters
                self.optimizer.step()
                # print(prof.key_averages().table(sort_by="cuda_time_total"))
            # log the trajectory
            if self.log_video and (epoch+1)%self.log_video_every==0:
                self.log_trajectory()
            # save the model
            if self.save:
                torch.save(self.model.state_dict(),'models/model_{0}.pt'.format(epoch))
        if self.log_video:
            self.log_trajectory()
