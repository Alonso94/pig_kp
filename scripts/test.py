from pig.agents.pig_agent import PIG_agent

import wandb


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
    'batch_size':8,
    'learning_rate':0.001,
    'num_keypoints':32,
    'padding':20,
    'epochs':200,
    'log_video':True,
    'log_video_every':20,
    'save_model':False,
    'batch_norm':True,
    # parameters for the pig loss
    'region_size':3,
    'bandwidth':0.001,
    'std_for_featuremap_generation':9,
    # 'threshold_for_featuremaps':0.0001,
    'masked_entropy_loss_weight':100.0,
    'overlapping_loss_weight':10.0,
    # parameters for the pcl loss
    'pcl_type': 'histogram', # 'histogram', 'representation' or 'learning'
    'num_samples':8,
    'margin_for_matrix_contrastive_loss':0.5,
    'contrastive_loss_weight':10.0,
    'matches_loss_weight':5.0,
    'non_matches_loss_weight':0.0,
    'sigma_for_mcl_soft':0.25,
    'dilation_kernel_size':7,
    'number_of_rings_around_the_keypoint':3,
    # hyperparameters for training the representation model
    "further_training": False,
    "num_epochs_for_representation_model":5,
    "lr_for_representation_model":0.0003,
    "weight_decay_for_representation_model":0.0000,
    "representation_size": 16,
    "matches_per_patch": 10,
    "noised_coords": False,
    # parameters for spatial consistency loss
    'spatial_consistency_loss_weight':1.0,
}

wandb.init(project="pcl_kp", name="pig-test", entity="irosa-ias", config=wandb.config,
              # mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()