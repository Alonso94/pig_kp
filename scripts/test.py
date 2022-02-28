from pig.agents.pig_agent import PIG_agent

import wandb

wandb.config={
    # parameters for dataset
    'tasks':'1,2',
    'number_of_demos':2,
    'number_of_stacked_frames':3,
    'width':320, # MIME
    'height':120, # MIME
    'channels':4, # RGB + 3 for depth
    # parameters for agent
    'batch_size':8,
    'learning_rate':0.001,
    'num_keypoints':64,
    'epochs':10,
    'log_video':True,
    'save_model':False,
    # parameters for the loss
    'region_size':3,
    'bandwidth':0.001,
    'std_for_featuremap_generation':5,
    'threshold_for_featuremaps':0.0001,
}

wandb.init(project="pig-test", name="pig-test", entity="3liyounes94", config=wandb.config,
              mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()