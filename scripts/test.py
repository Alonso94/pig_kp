from pig.agents.pig_agent import PIG_agent

import wandb


wandb.config={
    # parameters for dataset
    'tasks':'1,2',
    'number_of_demos':2,
    'number_of_stacked_frames':6,
    'width':320, # MIME 640
    'height':120, # MIME 240
    'channels':4, # RGB + 3 for depth
    # parameters for agent
    'batch_size':16,
    'learning_rate':0.001,
    'num_keypoints':16,
    'epochs':10,
    'log_video':True,
    'save_model':False,
    # parameters for the pig loss
    'region_size':3,
    'bandwidth':0.001,
    'std_for_featuremap_generation':3,
    'threshold_for_featuremaps':0.0001,
    'pig_loss_weight':10,
    # parameters for the pcl loss
    'num_samples':8,
    'margin_for_matrix_contrastive_loss':2,
    'contrastive_loss_weight':10,
    'matches_loss_weight':0.0,
    'non_matches_loss_weight':3,
    # 'sigma_for_mcl_soft':0.1,
    # parameters for spatial consistency loss
    'spatial_consistency_loss_weight':0.0,
}

wandb.init(project="pig-test", name="pig-test", entity="3liyounes94", config=wandb.config,
              # mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()