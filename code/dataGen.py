import os
import numpy as np
import h5py
import random as rn

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
np.random.seed(128)
rn.seed(128)
#-----------------------------------------#

def getMixConstant(speech, noise, snr):
    var = np.sum(np.power(speech, 2)) / speech.shape[0]
    varnoise = np.sum(np.power(noise, 2)) / noise.shape[0]
    ratio = np.power(10, snr / 10.0)
    sd = np.sqrt(var/varnoise/ratio)
    return sd

class DatasetContainer():
    def __init__(self, params, trainSplit, noise=False):
        self.noise = noise
        self.params = params
        self.dset = h5py.File(self.params['IN_PATH'], 'r')
        self.lmarkDSET = self.dset['landmarks']#[:,:,:]
        self.speechDSET = self.dset['speech']#[:,:]
        self.num_samples = self.lmarkDSET.shape[0]
        self.idxList = [(i, j) for i in range(self.num_samples) for j in range(75-self.params['NUMFRAMES']-1)]
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
        
        rn.shuffle(self.idxList)

        train_idxs = np.random.choice(len(self.idxList), int(len(self.idxList)*trainSplit), replace=False)
        self.trainlist = [self.idxList[i] for i in train_idxs]
        self.vallist= np.delete(self.idxList, train_idxs, axis=0)
        print('Number of train samples: %d, Number of val samples: %d' % (len(self.trainlist), len(self.vallist)))

    def getTrainDset(self):
        return FaceLandmarksDataset(self.trainlist, self.params, self.noise)

    def getValDset(self):
        return FaceLandmarksDataset(self.vallist, self.params, self.noise)

class FaceLandmarksDataset(Dataset):

    def __init__(self, idxlist, params, noise):
        self.noise = noise
        self.params = params
        self.dset = h5py.File(self.params['IN_PATH'], 'r')
        if self.noise:
            self.ndset = h5py.File(self.params['NOISE_PATH'], 'r')
            self.noise_list = []
            for key in self.ndset.keys():
                self.noise_list.append(self.ndset[key][:])
        # Landmark shapes: [samples, time_steps, 68*3]
        # speech shape: [samples, signal_length]
        self.lmarkDSET = self.dset['landmarks']#[:,:,:]
        self.speechDSET = self.dset['speech']#[:,:]
        self.idxList = idxlist
        self.augList = [-12, -9, -6, -3, 0, 3, 6]
        self.snrList = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        
    def __len__(self):
        return len(self.idxList)

    def __getitem__(self, idx):
        i, j = self.idxList[idx]

        rnd_dset = np.random.randint(0, high=5, size=[1, ])[0]
        rnd_dB = np.random.randint(0, high=len(self.augList), size=[1, ])[0]

        if self.noise:
            rnd_snr = np.random.randint(0, high=len(self.snrList), size=[1, ])[0]
            rnd_noise = np.random.randint(0, high=len(self.noise_list), size=[1, ])[0]
            noise = self.noise_list[rnd_noise]
            rnd_idx = np.random.randint(0, high=noise.shape[0]-self.params['INCREMENT']*self.params['NUMFRAMES'], size=[1, ])[0]
            noise = noise[rnd_idx:rnd_idx+self.params['INCREMENT']*self.params['NUMFRAMES']]

        cur_lmark = self.lmarkDSET[i, j:j+self.params['NUMFRAMES'], :]
        cur_speech = np.reshape(self.speechDSET[i, j*self.params['INCREMENT']:(j+self.params['NUMFRAMES'])*self.params['INCREMENT']],
                                [1, self.params['NUMFRAMES']*self.params['INCREMENT']])

        cur_speech = cur_speech*np.power(10.0, self.augList[rnd_dB]/20.0)

        noisy_speech = cur_speech
        if self.noise:
            sd = getMixConstant(cur_speech[0, :], noise, rnd_snr)
            noisy_speech = sd*noise + cur_speech          

        return noisy_speech, cur_speech, cur_lmark[self.params['MID'], :], cur_lmark[self.params['MID']-1, :]