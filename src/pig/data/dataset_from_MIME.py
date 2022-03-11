import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromMIME(Dataset):
    def __init__(self,config, demonstrator='human'):
        super(DatasetFromMIME,self).__init__()
        self.demonstrator=demonstrator.lower()

        self.width = config['width']
        self.height = config['height']

        self.number_of_demos=config['number_of_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']

        self.with_depth=config['with_depth']

        self.tasks=config['tasks'].split(',')

        self.visualize_preprocessing=False

        # load the dataset if it's already been created
        data_path='datasets/MIME_dataset_{0}x{1}_{2}tasks_{3}demos_{4}frames_depth({5}).pt'.format(self.width,self.height,len(self.tasks), self.number_of_demos,self.number_of_stacked_frames, self.with_depth)
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
        # frames=cv2.cvtColor(frames, cv2.COLOR_RGB2BGR)
        # show the frames
        cv2.imshow('sample',frames)
        cv2.waitKey(0)

    def sample_video_from_data(self, n_frames):
        # sample the index of the starting frame
        idx=np.random.randint(0,len(self.human_data)-n_frames)
        # find to which task the idx belongs by using the task_start_idx
        task=np.where(self.task_start_idx<=idx)[0][-1]
        # if the idx is less than task_start_idx[task]+n_frames then shift it
        if idx<self.task_start_idx[task]+n_frames:
            idx=self.task_start_idx[task]+n_frames
        # if idx is greater than the length of the task then shift it
        if idx>self.task_start_idx[task+1]:
            idx=self.task_start_idx[task+1]-1
        # The sample is n_frames frames ending at idx
        human_sample=self.human_data[idx-n_frames:idx]
        robot_sample=self.robot_data[idx-n_frames:idx]
        # form the sample
        sample={'human':human_sample,'robot':robot_sample}
        return sample

    def preprocess(self,frame,i):
        # apply bilateral filter to the frame
        bilateral=cv2.bilateralFilter(frame,9,150,150)
        # blur the frame
        smooth=cv2.blur(bilateral,(150,150))
        # divide the frame by the blurred frame
        division=cv2.divide(frame,smooth,scale=255)
        # sharpen the frame
        frame=cv2.addWeighted(division,2.0,cv2.GaussianBlur(frame,(0,0),5),-0.5,0)
        if i==0 and self.visualize_preprocessing:
            fig,axes=plt.subplots(2,2, constrained_layout=True)
            axes[0,0].imshow(bilateral)
            axes[0,0].set_title('bilateral filter')
            axes[0,1].imshow(smooth)
            axes[0,1].set_title('blur')
            axes[1,0].imshow(division)
            axes[1,0].set_title('divide')
            axes[1,1].imshow(frame)
            axes[1,1].set_title('sharpen')
            # remove the ticks
            for ax in axes.flat:
                ax.set(xticks=[],yticks=[])
            plt.show()
            # kill all the figures
            plt.close('all')
        return frame

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
            # preprocess the frame
            frame=self.preprocess(frame,i)
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
                if self.with_depth:
                    human_depth_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,human_videos[0]),depth=True)
                human_rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,human_videos[1]))
                if self.with_depth:
                    robot_depth_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,robot_videos[0]),depth=True)
                robot_rgb_frames=self.read_frames_from_video('MIME/{0}/{1}/{2}'.format(task,demo,robot_videos[1]))
                # unify the length of the frames
                if self.with_depth:
                    n_frames=min(human_depth_frames.shape[0],human_rgb_frames.shape[0],robot_depth_frames.shape[0],robot_rgb_frames.shape[0])
                else:
                    n_frames=min(human_rgb_frames.shape[0],robot_rgb_frames.shape[0])
                # add the task start index to the list
                self.task_start_idx.append(self.task_start_idx[-1]+n_frames)
                # cut the frames to the same length
                if self.with_depth:
                    human_depth_frames=human_depth_frames[:n_frames]
                human_rgb_frames=human_rgb_frames[:n_frames]
                if self.with_depth:
                    robot_depth_frames=robot_depth_frames[:n_frames]
                robot_rgb_frames=robot_rgb_frames[:n_frames]
                # concatenate the egb and depth frames
                if self.with_depth:
                    human_frames=np.concatenate((human_rgb_frames,human_depth_frames),axis=3)
                else:
                    human_frames=human_rgb_frames
                if self.with_depth:
                    robot_frames=np.concatenate((robot_rgb_frames,robot_depth_frames),axis=3)
                else:
                    robot_frames=robot_rgb_frames
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
