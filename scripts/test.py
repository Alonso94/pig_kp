from pig.agents.pig_agent import PIG_agent

import wandb

import torch
torch.autograd.set_detect_anomaly(True)

wandb.config={
    # parameters for dataset
    'tasks':'1,2',
    'number_of_demos':2,
    'number_of_stacked_frames':3,
    'with_depth':False,
    'width':320, # MIME 640 -> after crop to 320x240
    'height':240, # MIME 240
    'channels':3, # RGB + 3 for depth
    # parameters for agent
    'batch_size':4,
    'learning_rate':0.001,
    'num_keypoints':12,
    'padding':0,
    'epochs':200,
    'log_video':False,
    'log_video_every':50,
    'save_model':False,
    'batch_norm':True,
    # parameters for the pig loss
    'region_size':3,
    'bandwidth':0.001,
    'std_for_featuremap_generation':9,
    # 'threshold_for_featuremaps':0.0001,
    'masked_entropy_loss_weight':100.0,
    'overlapping_loss_weight':1.0,
    # parameters for the pcl loss
    'pcl_type': 'representation', # 'histogram', 'representation', 'learning_contrastive' or 'learning_AE'
    'num_samples':8,
    'margin_for_matrix_contrastive_loss':0.6,
    'contrastive_loss_weight':0.0,
    'matches_loss_weight':1.0,
    'non_matches_loss_weight':0.0,
    'sigma_for_mcl_soft':0.5,
    'dilation_kernel_size':7,
    'number_of_rings_around_the_keypoint':3,
    # hyperparameters for training the representation model
    "further_training": False,
    "representation_epochs": 50,
    "num_epochs_for_representation_model":5,
    "lr_for_representation_model":0.0003,
    "weight_decay_for_representation_model":0.0000,
    "representation_size": 16,
    "matches_per_patch": 10,
    "noised_coords": False,
    # parameters for spatial consistency loss
    'spatial_consistency_loss_weight':1.0,
    # status part of the loss
    'status_weight':0.3,
}

wandb.init(project="pig_test", name="pig+pcl_coords", entity="3liyounes94", config=wandb.config,
              mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()