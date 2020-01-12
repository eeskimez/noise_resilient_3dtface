import argparse
import math
import os
import random as rn
import shutil
import subprocess

import librosa
import numpy as np
from tqdm import tqdm
from plot_face import facePainter
from scipy.spatial import procrustes
import torch
import utils
from copy import deepcopy
from model import SPCH2FLM, SPCH2FLMTC

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-folder", type=str, help="input speech folder", required=True)
parser.add_argument("-m", "--model", type=str, help="Pre-trained model", required=True)
parser.add_argument("-o", "--out-fold", type=str, help="output folder", required=True)

parser.add_argument("--mean_shape", type=str, help="PCA mean shape vector npy file path", default="../data/mean_shape.npy")
parser.add_argument("--eigen_vectors", type=str, help="PCA eigen vectors npy file path", default="../data/eigen_vectors.npy")
parser.add_argument("--template_shape", type=str, help="PCA eigen vectors npy file path", default="../data/mean.npy")
parser.add_argument("-n", "--num-frames", type=int, help="Number of frames", default=7)
parser.add_argument("--temporal_condition", action="store_true")
parser.add_argument("--tcboost", type=str, help="Boost coefficients for autoregressive model", default="../data/tcboost.npy")

args = parser.parse_args()
pca_mean_vector = np.load(args.mean_shape)
pca_eigen_vectors = np.load(args.eigen_vectors)
if args.temporal_condition:
    pca_eigen_vectors_inv = np.linalg.pinv(pca_eigen_vectors)
    template_shape = np.load(args.template_shape)
    _, template_shape, _ = procrustes(np.reshape(pca_mean_vector,  template_shape.shape), template_shape)
    template_shape = np.reshape(template_shape, (1, 204))
    tcboost = np.load(args.tcboost)

def generateFace(root, filename):
    speech, sr = librosa.load(os.path.join(root, filename))
    speech = speech / np.max(np.abs(speech))
    speech_orig = speech[:]
    speech = librosa.resample(speech, sr, fs)

    increment = int(0.04*fs) # Increment rate for 25 FPS videos
    upper_limit = speech.shape[0]
    lower = 0
    predicted = np.zeros((0, pca_mean_vector.shape[1]))
    flag = 0

    # Pad zeros to start and end of the speech signal
    speech = np.insert(speech, 0, np.zeros((int(increment*num_frames/2))))
    speech = np.append(speech, np.zeros((int(increment*num_frames/2))))

    if args.temporal_condition:
        condition = np.dot(template_shape-pca_mean_vector, pca_eigen_vectors_inv)
        condition = np.reshape(condition, (1, pca_eigen_vectors.shape[0]))

    while True:
        cur_features = np.zeros((1, 1, num_frames*increment))
        local_speech = speech[lower:lower+num_frames*increment]
        
        if local_speech.shape[0] < num_frames*increment:
            local_speech = np.append(local_speech, np.zeros((num_frames*increment-local_speech.shape[0])))
            flag = 1
        
        cur_features[0, 0, :] = local_speech
        lower += increment
        if flag:
            break

        cur_features = torch.from_numpy(cur_features).float()
        if args.temporal_condition:
            condition = torch.from_numpy(condition).float()
            pred = model(cur_features, condition)[0].data.numpy()
            condition = deepcopy(pred[:, :])
            pred = pred * tcboost # Boost the predicted coefficients by multiplying them with hand-tuned coefficients
            # This way, the mouth movements can be exaggerated. You can play around by changing the boost coefficients.
            # In paper experiments, we haven't boosted the samples.
        else:
            pred = model(cur_features)[0].data.numpy()
        pred = (pca_mean_vector + np.dot(pred, pca_eigen_vectors))
        predicted = np.append(predicted, np.reshape(pred[0, :], (1, pca_mean_vector.shape[1])), axis=0)

    if len(predicted.shape) < 3:
        predicted = np.reshape(predicted, (predicted.shape[0], int(predicted.shape[1]/3), 3))

    # 2D video with painted face
    fp = facePainter(predicted, speech_orig, fs=sr)
    fp.paintFace(output_path, os.path.splitext(filename)[0]+'_painted')

    # 3D video with connected lines
    utils.write_video3D(predicted, speech_orig, sr, output_path, os.path.splitext(filename)[0]+'_3D', [-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], 25)

output_path = args.out_fold
num_frames = args.num_frames 
fs = 8000 # Sampling rate

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

params = {}
params['CUDA'] = False#torch.cuda.is_available()
params['DEVICE'] = torch.device("cuda" if params['CUDA'] else "cpu") 
params['kwargs'] = {'num_workers': 1, 'pin_memory': True} if params['CUDA'] else {}

if args.temporal_condition:
    model = SPCH2FLMTC().to(params['DEVICE'])
else:
    model = SPCH2FLM().to(params['DEVICE'])

test_folder = args.in_folder
model.load_state_dict(torch.load(args.model, map_location="cuda" if params['CUDA'] else "cpu"))

for root, dirs, files in os.walk(test_folder):
    for filename in files:
        if filename.endswith(('.WAV', '.wav', '.flac')):
            print (os.path.join(root, filename))
            generateFace(root, filename)
