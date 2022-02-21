from pig.agents.pig_agent import PIG_agent

import wandb

wandb.config={
    # parameters for dataset
    'tasks':'1,2',
    'number_of_demos':2,
    'number_of_stacked_frames':3,
    'width':240, # MIME
    'height':640, # MIME
    'channels':6, # RGB + 3 for depth
    # parameters for agent
    'batch_size':8,
    'learning_rate':0.001,
    'num_keypoints':16,
    'epochs':10,
    'log_video':True,
    'save_model':False,
}

wandb.init(project="pig-test", name="pig-test", entity="3liyounes94", config=wandb.config,
              mode="disabled"
            )

pig=PIG_agent(wandb.config)
pig.train()