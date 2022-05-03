from pig.agents.pig_agent_simitate import PIG_agent

import wandb

import torch
torch.autograd.set_detect_anomaly(True)

wandb.config={
    # parameters for dataset
    'tasks':'bring,rearrange',
    'number_of_demos':4,
    'number_of_stacked_frames':3,
    'with_depth':False,
    'width':480, # SIMITATE 960
    'height':270, # SIMITATE 540
    'channels':3, # RGB
    # parameters for agent
    'batch_size':4,
    'learning_rate':0.001,
    'num_keypoints':25,
    'palindrome':False,
    'padding':20,
    'epochs':50,
    'log_video':True,
    'log_video_every':5,
    'save_model':False,
    'batch_norm':True,
    # parameters for the pig loss
    'activation_score_threshold':10,
    'dynamic_score_threshold':25,
    'region_size':3,
    'bandwidth':0.001,
    'penalize_background':False,
    'std_for_featuremap_generation':16,
    'fm_threshold':0.3,
    'thresholded_fm_scale':3.5,
    'schedule': 0.1,
    'masked_entropy_loss_weight':80.0,
    'conditional_entropy_loss_weight':150.0,
    'active_overlapping_weight':1.5,
    'dynamic_overlapping_weight':0.8,
    'static_movement_loss_weight': 0.8,
    'dynamic_movement_loss_weight': 0.3,
    'status_weight':0.4,
    'mode_switching_weight':0.0,
    'activation_loss_weight':0.0,
}

wandb.init(project="pig_simitatae", name="pig_simitate", entity="3liyounes94", config=wandb.config,
              # mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()