import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import cv2

from pig.entropy_layer.entropy import Entropy

import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DatasetFromSIMITATE(Dataset):
    def __init__(self,config):
        super(DatasetFromSIMITATE,self).__init__()

        self.width = config['width']
        self.height = config['height']

        self.number_of_demos=config['number_of_demos']
        self.number_of_stacked_frames=config['number_of_stacked_frames']

        self.tasks=config['tasks'].split(',')

        self.visualize_preprocessing=True

        self.entropy_layer=Entropy(config['region_size'],config['bandwidth']).to(device)

        # load the dataset if it's already been created
        data_path='datasets/Simitate_dataset_{0}x{1}_{2}tasks_{3}demos_{4}frames.pt'.format(self.width,self.height,len(self.tasks), self.number_of_demos,self.number_of_stacked_frames)
        if os.path.exists(data_path):
            print('Loading dataset from',data_path)
            d = torch.load(data_path)
            self.data=d.data
            self.task_start_idx=d.task_start_idx
        else:
            print('Creating dataset from data')
            self.data=[]
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
        sample=self.data[idx-self.number_of_stacked_frames:idx]
        return sample

    def show_sample(self,idx):
        sample=self.__getitem__(idx)
        # stack the frames of the human and robot
        frames=np.concatenate(sample, axis=0)
        # use RGB to show the frames
        frames=frames.astype(np.uint8)
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
        sample=self.data[idx-n_frames:idx]
        return sample

    def preprocess1(self,frame):
        # # convert the frame to rgb
        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image=frame
        # apply bilateral filter to the frame
        bilateral=cv2.bilateralFilter(frame,12,150,150)
        # blur the frame
        smooth=cv2.blur(bilateral,(15,15))
        # divide the frame by the blurred frame
        division=cv2.divide(bilateral,smooth,scale=255)
        # sharpen the frame
        sharp=cv2.addWeighted(division,1.5,cv2.GaussianBlur(bilateral,(0,0),150),-0.7,0)
        frame=sharp
        # # # apply bilateral filter to the frame
        # frame=cv2.bilateralFilter(division,9,250,250)
        # frame=sharp 
        if self.visualize_preprocessing:
            fig,axes=plt.subplots(2,3, constrained_layout=True)
            axes[0,0].imshow(image)
            axes[0,0].set_title('input image')
            axes[0,1].imshow(bilateral)
            axes[0,1].set_title('bilateral')
            axes[0,2].imshow(smooth)
            axes[0,2].set_title('smooth')
            axes[1,0].imshow(division)
            axes[1,0].set_title('division')
            axes[1,1].imshow(sharp)
            axes[1,1].set_title('sharp')
            # convert the frame to a tensor
            frame_t=torch.from_numpy(image).float().to(device)
            frame_t=frame_t[None,None,:,:,:].permute(0,1,4,2,3)
            # pass the frame to the entropy layer
            entropy=self.entropy_layer(frame_t)
            # show the entropy
            axes[1,2].imshow(entropy[0,0,0].detach().cpu().numpy(),cmap='jet')
            axes[1,2].set_title('entropy')
            # remove the ticks
            for ax in axes.flat:
                ax.set(xticks=[],yticks=[])
            plt.show()
            # kill all the figures
            plt.close('all')
        return frame

    def preprocess(self,frame,i):
        # # convert the frame to rgb
        # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image=frame
        # blur the frame
        smooth=cv2.blur(frame,(15,15))
        # sharpen the frame
        sharp=cv2.addWeighted(frame,2.5,smooth,-1.2,0) 
        # divide the frame by the blurred frame
        division=cv2.divide(sharp,smooth,scale=255)
        frame=division
        # # # apply bilateral filter to the frame
        # frame=cv2.bilateralFilter(division,9,250,250)
        # frame=sharp 
        if i==90 and self.visualize_preprocessing:
            fig,axes=plt.subplots(2,3, constrained_layout=True)
            axes[0,0].imshow(image)
            axes[0,0].set_title('input image')
            axes[0,1].imshow(smooth)
            axes[0,1].set_title('smooth')
            axes[0,2].imshow(sharp)
            axes[0,2].set_title('sharp')
            axes[1,0].imshow(division)
            axes[1,0].set_title('division')
            axes[1,1].imshow(frame)
            axes[1,1].set_title('-')
            # convert the frame to a tensor
            frame_t=torch.from_numpy(image).float().to(device)
            frame_t=frame_t[None,None,:,:,:].permute(0,1,4,2,3)
            # pass the frame to the entropy layer
            entropy=self.entropy_layer(frame_t,sharp,division)
            # show the entropy
            axes[1,2].imshow(entropy[0,0,0].detach().cpu().numpy(),cmap='jet')
            axes[1,2].set_title('entropy')
            # remove the ticks
            for ax in axes.flat:
                ax.set(xticks=[],yticks=[])
            plt.show()
            # kill all the figures
            plt.close('all')
        return frame

    def read_frames_from_video(self,path):
        # list all the frames in the path
        file_names=os.listdir(path)
        # sort the frames by their names
        file_names.sort()
        # create a list to store the frames
        frames=[]
        for i,name in enumerate(file_names):
            # read the frame
            frame = cv2.imread("{0}/{1}".format(path,name))
            # # center crop the frame from 640x240 to 320x240
            # frame=frame[:,120:480,:]
            if self.width!=None and self.height!=None:
                # resize the frame
                frame=cv2.resize(frame,(self.width,self.height))
            # preprocess the frame
            # frame=self.preprocess(frame,i)
            frames.append(frame)
        # convert the list to a numpy array
        frames=np.array(frames)
        # return the frames
        return frames

    def read_frames(self):
        # iterate over the tasks
        for task in self.tasks:
            for i in range(self.number_of_demos):
                frames=self.read_frames_from_video('SIMITATE/{0}_{1}'.format(task,i+1))
                # The length of the frames
                n_frames=frames.shape[0]
                # add the task start index to the list
                self.task_start_idx.append(self.task_start_idx[-1]+n_frames)
                # append the frames to the list
                self.data.append(frames)
        # convert the lists to numpy arrays
        self.data=np.concatenate(self.data, axis=0)
        self.task_start_idx=np.array(self.task_start_idx)
        print('data shape:',self.data.shape)
        print('task_start_idx:',self.task_start_idx)