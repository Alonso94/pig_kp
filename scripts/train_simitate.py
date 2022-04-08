from pig.agents.pig_agent_simitate import PIG_agent

import wandb

import torch
torch.autograd.set_detect_anomaly(True)

wandb.config={
    # parameters for dataset
    'tasks':'bring,rearrange',
    'number_of_demos':4,
    'number_of_stacked_frames':6,
    'with_depth':False,
    'width':480, # SIMITATE 960 -> after crop to 320x240
    'height':270, # SIMITATE 540
    'channels':3, # RGB
    # parameters for agent
    'batch_size':4,
    'learning_rate':0.001,
    'num_keypoints':32,
    'palindrome':False,
    'padding':20,
    'epochs':200,
    'log_video':True,
    'log_video_every':20,
    'save_model':False,
    'batch_norm':True,
    # parameters for the pig loss
    'activation_score_threshold':15,
    'region_size':3,
    'bandwidth':0.001,
    'penalize_background':False,
    'std_for_featuremap_generation':9,
    'fm_threshold':0.1,
    'schedule': 0.2,
    'masked_entropy_loss_weight':100.0,
    'conditional_entropy_loss_weight':50.0,
    'overlapping_loss_weight':10.0,
    'movement_loss_weight':0.1,
    'palindrome_weight':0.0,
    'status_weight':1.0,
}

wandb.init(project="pig_imitatae", name="pig_simitate", entity="3liyounes94", config=wandb.config,
              mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()