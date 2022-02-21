from pig.data.dataset_from_data import DatasetFromData

import wandb

wandb.config={
    'tasks':'1,2',
    'number_of_demos':2,
    'number_of_stacked_frames':3,
    'width':None,
    'height':None
}

d=DatasetFromData(wandb.config)
d.show_sample(100)