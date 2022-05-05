from builtins import print
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

from pig.entropy_layer.entropy import Entropy

import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromMIME(Dataset):
    def __init__(self,config):
        super(DatasetFromMIME,self).__init__()

        self.width = config['width']
        self.height = config['height']

        self.training_demos=config['training_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']
        self.evaluation_demos=config['evaluation_demos']

        self.tasks=config['tasks'].split(',')

        # load the dataset if it's already been created
        data_path='datasets/MIME_dataset_{0}x{1}_{2}tasks_{3}training_{4}validation_{5}frames.pt'.format(self.width,self.height,len(self.tasks), self.training_demos, self.evaluation_demos,self.number_of_stacked_frames)
        if os.path.exists(data_path):
            print('Loading dataset from',data_path)
            d = torch.load(data_path)
            self.data=d.data
            self.demo_start_idx=d.demo_start_idx
            self.stats=d.stats
        else:
            print('Creating dataset from data')
            self.data=[]
            self.demo_start_idx=[0]
            self.stats=[]
            self.read_frames()
            # create a folder to store the dataset
            os.makedirs('datasets',exist_ok=True)
            torch.save(self, data_path)

    def __len__(self):
        # the length is the last element of the task_start_idx list
        return self.demo_start_idx[-self.evaluation_demos]

    def __getitem__(self, idx):
        # find to which task the idx belongs by using the task_start_idx
        task=np.where(self.demo_start_idx<=idx)[0][-1]
        # if the idx is less than task_start_idx[task]+self.number_of_stacked_frames then shift it
        if idx<self.demo_start_idx[task]+self.number_of_stacked_frames:
            idx=self.demo_start_idx[task]+self.number_of_stacked_frames
        # The sample is number_of_stacked_frames frames ending at idx
        samlpe=self.data[idx-self.number_of_stacked_frames:idx]
        return samlpe

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        # stack the frames of the human and robot
        frames=np.concatenate(sample, axis=0)
        # use RGB to show the frames
        frames=frames[:,:,:3].astype(np.uint8)
        # frames=cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        # show the frames
        cv2.imshow('sample',frames)
        cv2.waitKey(0)

    def sample_video_from_data(self, task_idx=None):
        # sample the index of the starting frame
        if task_idx is None:
            task_idx=np.random.randint(0,self.training_demos+self.evaluation_demos-1)
        else:
            task_idx=task_idx%(self.training_demos+self.evaluation_demos)
        start_idx=self.demo_start_idx[task_idx]
        end_idx=self.demo_start_idx[task_idx+1]
        # The sample is n_frames frames ending at idx
        sample=self.data[start_idx:end_idx]
        return sample

    def read_frames_from_video(self,video_path, depth=False):
        # read the video
        video=cv2.VideoCapture(video_path)
        # get the number of frames
        n_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # get the frames
        frames=[]
        for i in range(n_frames):
            # read the frame
            ret, frame = video.read()
            # center crop the frame from 640x240 to 320x240
            frame=frame[:,120:480,:]
            if self.width!=None and self.height!=None:
                # resize the frame
                frame=cv2.resize(frame,(self.width,self.height))
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
            # iterate over training demos from this task
            for i in range(self.training_demos):
                demo=demos[i]
                # list of the videos for this demo
                videos=os.listdir('MIME/{0}/{1}'.format(task,demo))
                # the videos whose name starts with 'hd'
                videos=sorted([video for video in videos if video.startswith('hd')])
                rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,videos[1]))
                # the length of the video
                n_frames=rgb_frames.shape[0]
                # add the task start index to the list
                self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
                # append the frames to the list
                self.data.append(rgb_frames)
        for task in self.tasks:
            # list of the demos for this task
            demos=os.listdir('MIME/{0}'.format(task))
            # iterate over evaluation demos from this task
            for i in range(self.training_demos,self.evaluation_demos):
                demo=demos[i]
                # list of the videos for this demo
                videos=os.listdir('MIME/{0}/{1}'.format(task,demo))
                # the videos whose name starts with 'hd'
                videos=sorted([video for video in videos if video.startswith('hd')])
                rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,videos[1]))
                # the length of the video
                n_frames=rgb_frames.shape[0]
                # add the task start index to the list
                self.demo_start_idx.append(self.demo_start_idx[-1]+n_frames)
                # append the frames to the list
                self.data.append(rgb_frames)
        # convert the lists to numpy arrays
        self.data=np.concatenate(self.data, axis=0)
        self.demo_start_idx=np.array(self.demo_start_idx)
        # compute the mean and the std of the dataset for each channel
        self.stats=[np.mean(self.data,axis=(0,1,2)),np.std(self.data,axis=(0,1,2))]
        print('data shape:',self.data.shape)
        print('task_start_idx:',self.demo_start_idx)
        print('stats:',self.stats)
