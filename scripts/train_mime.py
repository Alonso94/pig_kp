from pig.agents.pig_agent_mime import PIG_agent

import wandb

import torch
torch.autograd.set_detect_anomaly(True)

wandb.config={
    'model_name':'ME + status + movement + inactive overlapping',
    # parameters for dataset
    'tasks':'1,2',
    'training_demos':4,
    'evaluation_demos':2,
    'number_of_stacked_frames':3,
    'width':320, # MIME 640 -> after crop to 320x240
    'height':240, # MIME 240
    'channels':3, # RGB
    # parameters for agent
    'batch_size':4,
    'learning_rate':0.001,
    'num_keypoints':25,
    'padding':20,
    'epochs':50,
    'log_video':True,
    'log_video_every':5,
    'save_model':True,
    'batch_norm':True,
    # parameters for the pig loss
    'activation_score_threshold':10,
    'region_size':3,
    'bandwidth':0.001,
    'std_for_featuremap_generation':16,
    'fm_threshold':0.1,
    'thresholded_fm_scale':3.5,
    'masked_entropy_loss_weight':100.0,
    'conditional_entropy_loss_weight':0.0,
    'active_overlapping_weight':0.0,
    'dynamic_overlapping_weight':0.0,
    'inactive_overlapping_weight':0.1,
    'static_movement_loss_weight': 0.5,
    'dynamic_movement_loss_weight': 0.2,
    'status_weight':0.3,
}

wandb.init(project="MIEL", name=wandb.config['model_name'], entity="irosa-ias", config=wandb.config,
              # mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()
for i in range(6):
  pig.eval()