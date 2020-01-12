import argparse
import json
import math
import os
import random as rn
import shutil
import sys
from collections import defaultdict

import h5py
import numpy as np
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from dataGen import DatasetContainer
from model import SPCH2FLM, SPCH2FLMTC
import utils
# Change this to specify GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
os.environ['PYTHONHASHSEED'] = '128'
np.random.seed(128)
rn.seed(128)
torch.manual_seed(128)
#-----------------------------------------#

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-file", type=str, help="Input file containing train data", default=None)
    parser.add_argument("-n", "--in-noise", type=str, help="Input file containing noise data", default=None)
    parser.add_argument("-m", "--model", type=str, help="DNN model to use", default=None)
    parser.add_argument("--temporal_condition", action="store_true")
    parser.add_argument("-o", "--out-fold", type=str, help="output folder", default='../models/def')
    args = parser.parse_args()

    params = {}
    params['LEARNING_RATE'] = 1e-04
    params['BATCHSIZE'] = 128
    params['NUMFRAMES'] = 7
    params['INCREMENT'] = int(0.04*8000)
    params['MID'] = int(math.floor(params['NUMFRAMES']/2.0))
    params['LMARKDIM'] = 3
    params['OUT_SHAPE'] = 6
    params['IN_PATH'] = args.in_file
    params['OUT_PATH'] = args.out_fold
    params['NUM_IT'] = int(2108771/params['BATCHSIZE'])
    params['NUM_EPOCH'] = 100
    params['ENV_NAME'] = 'SPCH2FLM'
    params['M_PATH'] = args.model
    params['NOISE_PATH'] = args.in_noise
    params['TC'] = args.temporal_condition
   
    if not os.path.exists(params['OUT_PATH']):
        os.makedirs(params['OUT_PATH'])
    else:
        shutil.rmtree(params['OUT_PATH'])
        os.mkdir(params['OUT_PATH'])

    if not os.path.exists(os.path.join(params['OUT_PATH'], 'inter')):
        os.makedirs(os.path.join(params['OUT_PATH'], 'inter'))
    else:
        shutil.rmtree(os.path.join(params['OUT_PATH'], 'inter'))
        os.mkdir(os.path.join(params['OUT_PATH'], 'inter'))

    with open(os.path.join(params['OUT_PATH'], 'params.txt'), 'w') as file:
        file.write(json.dumps(params, sort_keys=True, separators=('\n', ':')))

    params['CUDA'] = torch.cuda.is_available()
    print('Cuda device available: ', params['CUDA'])
    params['DEVICE'] = torch.device("cuda" if params['CUDA'] else "cpu") 
    params['kwargs'] = {'num_workers': 0, 'pin_memory': True} if params['CUDA'] else {}

    return params

def train():
    params = initParams()
    dataset = DatasetContainer(params, 0.92, noise=params['NOISE_PATH'])
    train_dset = dataset.getTrainDset()
    val_dset = dataset.getValDset()

    if params['TC']:
        model = SPCH2FLMTC().to(params['DEVICE'])
    else:
        model = SPCH2FLM().to(params['DEVICE'])
    model.apply(init_weights)
    
    if params['M_PATH']:
        model.load_state_dict(torch.load(os.path.join(params['M_PATH'], 'SPCH2FLM.pt'), map_location="cuda" if params['CUDA'] else "cpu"), strict=True)

    optimizer = optim.Adam(model.parameters(), lr=params['LEARNING_RATE'], betas=(0.5, 0.999))

    train_loader = torch.utils.data.DataLoader(train_dset,
                                               batch_size=params['BATCHSIZE'], 
                                               shuffle=True, 
                                               **params['kwargs'])

    val_loader = torch.utils.data.DataLoader(val_dset,
                                               batch_size=params['BATCHSIZE'], 
                                               shuffle=True, 
                                               **params['kwargs'])

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    total_iter = 0 
    prev_loss = np.inf
    for epoch in tqdm(range(params['NUM_EPOCH'])):
        model.train()
        lossDict = defaultdict(list)
        diterator = iter(train_loader)
        with trange(len(train_loader)) as t:  
            for i in t:
                (noisy, clean, w, c_w) = next(diterator)
                if params['NOISE_PATH']:
                    noisy = noisy.to(params['DEVICE'])
                clean, w, c_w = clean.to(params['DEVICE']), w.to(params['DEVICE']), c_w.to(params['DEVICE'])
                optimizer.zero_grad()
                
                if params['TC']:
                    clean_w_p, clean_features = model(clean, c_w)
                else:
                    clean_w_p, clean_features = model(clean)
                clean_loss = l1_loss(clean_w_p, w)
                loss = clean_loss

                if params['NOISE_PATH']:
                    if params['TC']:
                        noisy_w_p, noisy_features = model(noisy, c_w)
                    else:
                        noisy_w_p, noisy_features = model(noisy)
                    noisy_loss = l1_loss(noisy_w_p, w)
                    feature_loss = l2_loss(clean_features, noisy_features)
                    loss += noisy_loss + 1e-3*feature_loss
                
                lossDict['loss'].append(loss.item())
                if params['NOISE_PATH']:
                    lossDict['f_loss'].append(feature_loss.item())
                    lossDict['c_loss'].append(clean_loss.item())
                    lossDict['n_loss'].append(noisy_loss.item())
                loss.backward()
                optimizer.step()
                
                desc_str = ''
                for key in sorted(lossDict.keys()):
                    desc_str += key + ': %.5f' % (np.nanmean(lossDict[key])) + ', '
                t.set_description(desc_str)
                # t.set_description("loss: %.5f, cur_loss: %.5f" % (np.mean(lossDict['abs']), lossDict['abs'][-1]))

                total_iter+=1
        
        model.eval()
        diterator = iter(val_loader)
        with trange(len(val_loader)) as t: 
            for i in t:
                (noisy_data, data, target, condition) = next(diterator)
                data, target, condition = data.to(params['DEVICE']), target.to(params['DEVICE']), condition.to(params['DEVICE'])
                if params['NOISE_PATH']:
                    data = noisy_data.to(params['DEVICE'])
                if params['TC']:
                    output, _ = model(data, condition)
                else:
                    output, _ = model(data)
                loss = F.l1_loss(output, target, reduce=True)
                lossDict['abs_val'].append(loss.item())
                t.set_description("loss: %.5f, cur_loss: %.5f" % (np.mean(lossDict['abs_val']), lossDict['abs_val'][-1]))

        trainLoss = np.mean(lossDict['abs'])
        valLoss = np.mean(lossDict['abs_val'])

        cur_loss = np.mean(lossDict['abs_val'])
        print("Epoch loss: ", cur_loss)
        if prev_loss > cur_loss:
            print("Loss has improved from %.5f to %.5f" % (prev_loss, cur_loss))
            prev_loss = cur_loss
            torch.save(model.state_dict(), os.path.join(params['OUT_PATH'], 'SPCH2FLM.pt'))
            print("Model is saved to: ", os.path.join(params['OUT_PATH'], 'SPCH2FLM.pt'))
            early_stopping_cnt = 0
        else:
            early_stopping_cnt += 1
            torch.save(model.state_dict(), os.path.join(params['OUT_PATH'], 'inter','SPCH2FLM.pt'))
            print("Early stopping counter:", early_stopping_cnt)

        if early_stopping_cnt == 20:
            break

if __name__ == "__main__":
    train()
