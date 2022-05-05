from pig.data.dataset_from_MIME import DatasetFromMIME
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
import os

import warnings
warnings.simplefilter("ignore", UserWarning)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PIG_agent(nn.Module):
    def __init__(self,config):
        super().__init__()
        # load the dataset
        self.dataset=DatasetFromMIME(config)
        # self.dataset.show_sample(100)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # initialize the model
        self.model=Encoder(config, self.dataset.stats).to(device)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        # initialize the pig loss
        self.pig_loss=PatchInfoGainLoss(config)
        # initialize the patch extractor
        self.patch_extractor=PatchExtractor(config).to(device)
        # initialize the wandb
        wandb.watch(self.model,log_freq=1000)
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer(config)
        self.log_video=config['log_video']
        self.log_video_every=config['log_video_every']
        self.save=config['save_model']
        if self.save:
            # create a folder to store the model
            os.makedirs('models',exist_ok=True)
        self.epochs=config['epochs']
        self.num_keypoints=config['num_keypoints']
        self.model_name=config['model_name']
        self.evaluation_counter=0
        print('Model initialized')

    def log_trajectory(self, epoch):
        # get the data
        sample=self.dataset.sample_video_from_data()
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device).unsqueeze(0)
                kp=self.model(data)
                coords[0,i:i+10]=kp[...,:2]
                active_status[0,i:i+10]=kp[...,2]
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Epoch {0}'.format(epoch))
        self.model.train()
    
    def eval(self):
        self.evaluation_counter+=1
        # load the model
        self.model.load_state_dict(torch.load('models/{0}.pt'.format(self.model_name)))
        # get the data
        sample=self.dataset.sample_video_from_data(task_idx=self.evaluation_counter-1)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            coords=torch.zeros(1,sample.shape[0],self.num_keypoints,2)
            active_status=torch.zeros(1,sample.shape[0],self.num_keypoints)
            for i in range(0,sample.shape[0],10):
                data=torch.tensor(sample[i:i+10]).float().permute(0,3,1,2).to(device).unsqueeze(0)
                kp=self.model(data)
                coords[0,i:i+10]=kp[...,:2]
                active_status[0,i:i+10]=kp[...,2]
            self.visualizer.log_video(sample[...,:3],coords, active_status, label='Evaluation {0}'.format(self.evaluation_counter))

    def train(self):
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            # for sample in self.dataloader:
            for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # permute the data and move them to the device, enable the gradients
                data=sample.float().permute(0,1,4,2,3).to(device)
                # get the output
                kp=self.model(data)
                coords=kp[...,:2]
                active_status=kp[...,2]
                # generate feature maps around keypoints
                feature_maps=self.patch_extractor(coords, data.shape[-2:])
                # compute the loss
                loss=self.pig_loss(coords.clone(), feature_maps.clone(), active_status, data)
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                # update the parameters
                self.optimizer.step()
            # log the trajectory
            if self.log_video and (epoch+1)%self.log_video_every==0:
                self.log_trajectory(epoch=epoch)
        # save the model
        if self.save:
            torch.save(self.model.state_dict(),'models/{0}.pt'.format(self.model_name))
        if self.log_video:
            self.log_trajectory(self.epochs)
