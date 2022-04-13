from pig.data.dataset_from_MIME import DatasetFromMIME
from pig.models.kpae import Encoder
from pig.utils.extract_patches import PatchExtractor
from pig.utils.trajectory_visualization import TrajectoryVisualizer
from pig.losses.pig import PatchInfoGainLoss
from pig.losses.scl import SpatialConsistencyLoss

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
        self.dataset=DatasetFromMIME(config)
        # self.dataset.show_sample(530)
        # initialize the dataloaer
        self.dataloader=DataLoader(self.dataset,batch_size=config['batch_size'],shuffle=True)
        # initialize the model
        self.model=Encoder(config).to(device)
        # initialize the optimizer
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        # initialize the pig loss
        self.pig_loss=PatchInfoGainLoss(config)
        # initliaze the pcl loss
        self.learn_representation=False
        if config['pcl_type']=='histogram':
            from pig.losses.pcl_histogram import PatchContrastiveLoss
        if config['pcl_type']=='representation':
            from pig.losses.pcl_representation import PatchContrastiveLoss
        if config['pcl_type']=='learning_contrastive':
            from pig.losses.pcl_learning_contrastive import PatchContrastiveLoss
            self.learn_representation=True
        if config['pcl_type']=='learning_AE':
            from pig.losses.pcl_learning_AE import PatchContrastiveLoss
            self.learn_representation=True
        if config['pcl_type']=='pcl_coords':
            from pig.losses.pcl_coords import PatchContrastiveLoss
            self.learn_representation=False
        self.pcl_loss=PatchContrastiveLoss(config)
        # initialize the spatial consistency loss
        self.scl_loss=SpatialConsistencyLoss(config)
        # initialize the patch extractor
        self.patch_extractor=PatchExtractor(config, std=config['std_for_featuremap_generation']).to(device)
        # initialize the wandb
        wandb.watch(self.model,log_freq=1000)
        # initialize the trajectory visualizer
        self.visualizer=TrajectoryVisualizer()
        self.log_video=config['log_video']
        self.log_video_every=config['log_video_every']
        self.save=config['save_model']
        self.epochs=config['epochs']
        self.representation_epochs=config['representation_epochs']
        self.palindrome=config['palindrome']
        self.palindrome_weight=config['palindrome_weight']
        self.palindrome_loss=nn.MSELoss()
        # self.log_trajectory()
        # input()

    def log_trajectory(self):
        # get the data
        sample=self.dataset.sample_video_from_data(100)
        # freeze the encoder
        self.model.eval()
        with torch.no_grad():
            human_data=torch.tensor(sample['human']).float().permute(0,3,1,2).to(device).unsqueeze(0)
            kp=self.model(human_data)
            coords=kp[...,:2]
            active_status=kp[...,2]
            dynamic_status=kp[...,3]
            self.visualizer.log_video(sample['human'][...,:3],coords, active_status, dynamic_status,'human')
            # robot_data=torch.tensor(sample['robot']).float().permute(0,3,1,2).to(device).unsqueeze(0)
            # coords=self.model(robot_data)
            # self.visualizer.log_video(sample['robot'][...,:3],coords,'robot')
        # unfreeze the encoder
        self.model.train()

    def train(self):
        # self.log_trajectory()
        if self.learn_representation:
            for epoch in trange(self.representation_epochs, desc="Training the representation model"):
                for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                    # get the data
                    human_data=sample['human']
                    robot_data=sample['robot']
                    # Training the model using human data
                    # permute the data and move them to the device, enable the gradients
                    human_data=human_data.float().permute(0,1,4,2,3).to(device)
                    # generate random coordinates (integers between 0 and the size of the image)
                    random_x=torch.randint(0,human_data.shape[3],(*human_data.shape[:2],1),dtype=torch.int64)
                    random_y=torch.randint(0,human_data.shape[2],(*human_data.shape[:2],1),dtype=torch.int64)
                    # concatenate the random coordinates
                    coords=torch.cat((random_x,random_y),dim=1)
                    # generate feature maps around keypoints
                    feature_maps=self.patch_extractor(coords,human_data.shape[-2:])
                    self.pcl_loss.train_representation(coords.clone(),feature_maps,human_data)
        # train the model
        # for epoch in range(self.epochs):
        for epoch in trange(self.epochs, desc="Training the model"):
            # for sample in self.dataloader:
            for sample in tqdm(self.dataloader,desc='Epoch {0}'.format(epoch), leave=False):
                # get the data
                human_data=sample['human']
                robot_data=sample['robot']
                # Training the model using human data
                # permute the data and move them to the device, enable the gradients
                human_data=human_data.float().permute(0,1,4,2,3).to(device)
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                # if self.palindrome:
                #     flipped_data=human_data.flip(1)
                #     human_data=torch.cat((human_data,flipped_data),dim=1)
                # get the output
                kp=self.model(human_data)
                coords=kp[...,:2]
                active_status=kp[...,2]
                dynamic_status=kp[...,3]
                # coords.register_hook(lambda grad: print("coords grad",grad.mean()))
                # status.register_hook(lambda grad: print("status grad",grad.mean()))
                # generate feature maps around keypoints
                feature_maps=self.patch_extractor(coords, human_data.shape[-2:])
                # compute the loss
                loss=0
                # loss+=self.scl_loss(coords1.clone())
                # loss+=self.pcl_loss(coords.clone(),feature_maps.clone(), status, human_data)
                loss+=self.pig_loss(coords.clone(), feature_maps.clone(), active_status, dynamic_status, human_data)
                # if self.palindrome:
                #     flipped_coords=coords.flip(1)
                #     loss+= self.palindrome_weight * self.palindrome_loss(coords,flipped_coords)
                #     wandb.log({'palindrome_loss': self.palindrome_loss(coords,flipped_coords).item()})
                # compute the gradients
                self.optimizer.zero_grad()
                loss.backward()
                # update the parameters
                self.optimizer.step()
                # log the loss
                # wandb.log({'loss':loss.item()})
                # print(prof.key_averages().table(sort_by="cuda_time_total"))
                # input()
                # # Training the model using robot data
                # robot_data=robot_data.float().permute(0,1,4,2,3).to(device)
                # coords2=self.model(robot_data)
                # # generate feature maps around keypoints
                # feature_maps=self.patch_extractor(coords2,robot_data.shape[-2:])
                # loss=0
                # # loss+=self.scl_loss(coords2.clone())
                # # loss+=self.pcl_loss(feature_maps,robot_data)
                # loss+= self.pig_loss(feature_maps,robot_data)
                # # compute the gradients
                # self.optimizer.zero_grad()
                # loss.backward()
                # # input()
                # # update the parameters
                # self.optimizer.step()
                # # log the loss
                # wandb.log({'loss':loss.item()})
            # log the trajectory
            if self.log_video and epoch%self.log_video_every==0:
                self.log_trajectory()
            # save the model
            if self.save:
                torch.save(self.model.state_dict(),'models/model_{0}.pt'.format(epoch))
        if self.log_video:
            self.log_trajectory()
