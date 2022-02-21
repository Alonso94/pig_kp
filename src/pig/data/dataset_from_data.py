import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromData(Dataset):
    def __init__(self,config, demonstrator='human'):
        super(DatasetFromData,self).__init__()
        self.demonstrator=demonstrator.lower()

        self.width = config['width']
        self.height = config['height']

        self.number_of_demos=config['number_of_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']

        self.tasks=config['tasks'].split(',')

        # load the dataset if it's already been created
        data_path='datasets/dataset_{0}tasks_{1}demos_{2}frames.pt'.format(len(self.tasks), self.number_of_demos,self.number_of_stacked_frames)
        if os.path.exists(data_path):
            print('Loading dataset from',data_path)
            d = torch.load(data_path)
            self.human_data=d.human_data
            self.robot_data=d.robot_data
            self.task_start_idx=d.task_start_idx
        else:
            print('Creating dataset from data')
            self.human_data=[]
            self.robot_data=[]
            self.task_start_idx=[0]
            self.read_frames()
            # create a folder to store the dataset
            os.makedirs('datasets',exist_ok=True)
            torch.save(self, data_path)

    def __len__(self):
        # the length is the last element of the task_start_idx list
        return self.task_start_idx[-1]

    def __getitem__(self, idx):
        # find to which task the idx belongs by using the task_start_idx
        task=np.where(self.task_start_idx<=idx)[0][-1]
        # if the idx is less than task_start_idx[task]+self.number_of_stacked_frames then shift it
        if idx<self.task_start_idx[task]+self.number_of_stacked_frames:
            idx=self.task_start_idx[task]+self.number_of_stacked_frames
        # The sample is number_of_stacked_frames frames ending at idx
        human_sample=self.human_data[idx-self.number_of_stacked_frames:idx]
        robot_sample=self.robot_data[idx-self.number_of_stacked_frames:idx]
        # form the sample
        sample={'human':human_sample,'robot':robot_sample}
        return sample

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        # stack the frames of the human and robot
        human_frames=np.concatenate(sample['human'], axis=0)
        robot_frames=np.concatenate(sample['robot'], axis=0)
        # stack them to one image, two rows
        frames=np.concatenate((human_frames,robot_frames),axis=1)
        # use RGB to show the frames
        frames=frames[:,:,:3].astype(np.uint8)
        frames=cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        # show the frames
        cv2.imshow('sample',frames)
        cv2.waitKey(0)

    def read_frames_from_video(self,video_path):
        # read the video
        video=cv2.VideoCapture(video_path)
        # get the number of frames
        n_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # get the frames
        frames=[]
        for i in range(n_frames):
            # read the frame
            ret, frame = video.read()
            if self.width!=None and self.height!=None:
                # resize the frame
                frame=cv2.resize(frame,(self.width,self.height))
            # add the frame to the list
            frames.append(frame)
        # close the video
        video.release()
        # convert the list to a numpy array
        frames=np.array(frames)
        # return the frames
        return frames

    def read_frames(self):
        # iterate over the tasks
        for task in self.tasks:
            # list of the demos for this task
            demos=os.listdir('MIME/{0}'.format(task))
            # iterate over a number of demos from this task
            for demo in demos[:self.number_of_demos]:
                # list of the videos for this demo
                videos=os.listdir('MIME/{0}/{1}'.format(task,demo))
                # concatenate the videos whose name starts with 'hd' and 'rd'
                human_videos=sorted([video for video in videos if video.startswith('hd')])
                robot_videos=sorted([video for video in videos if video.startswith('rd')])
                # read the frames
                human_depth_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,human_videos[0]))
                human_rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,human_videos[1]))
                robot_depth_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,robot_videos[0]))
                robot_rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,robot_videos[1]))
                # unify the length of the frames
                n_frames=min(human_depth_frames.shape[0],human_rgb_frames.shape[0],robot_depth_frames.shape[0],robot_rgb_frames.shape[0])
                # add the task start index to the list
                self.task_start_idx.append(self.task_start_idx[-1]+n_frames)
                # cut the frames to the same length
                human_depth_frames=human_depth_frames[:n_frames]
                human_rgb_frames=human_rgb_frames[:n_frames]
                robot_depth_frames=robot_depth_frames[:n_frames]
                robot_rgb_frames=robot_rgb_frames[:n_frames]
                # concatenate the egb and depth frames
                human_frames=np.concatenate((human_rgb_frames,human_depth_frames),axis=3)
                robot_frames=np.concatenate((robot_rgb_frames,robot_depth_frames),axis=3)
                # append the frames to the list
                self.human_data.append(human_frames)
                self.robot_data.append(robot_frames)
        # convert the lists to numpy arrays
        self.human_data=np.concatenate(self.human_data, axis=0)
        self.robot_data=np.concatenate(self.robot_data, axis=0)
        self.task_start_idx=np.array(self.task_start_idx)
        print('human data shape:',self.human_data.shape)
        print('robot data shape:',self.robot_data.shape)
        print('task_start_idx:',self.task_start_idx)
