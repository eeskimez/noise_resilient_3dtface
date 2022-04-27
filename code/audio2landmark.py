import argparse
import os
import shutil
from pathlib import Path
from time import clock

import librosa
import numpy as np
import torch

from model import SPCH2FLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-i", "--in-file", type=str, help="input speech file", required=True)
parser.add_argument("-o", "--out-fold", type=str, help="output folder", required=True)
parser.add_argument("-m", "--model", type=str, help="Pre-trained model", default="../pre_trained/1D_CNN.pt")
parser.add_argument("--mean_shape", type=str, help="PCA mean shape vector npy file path", default="../data/mean_shape.npy")
parser.add_argument("--eigen_vectors", type=str, help="PCA eigen vectors npy file path", default="../data/eigen_vectors.npy")
parser.add_argument("-n", "--num-frames", type=int, help="Number of frames", default=7)

args = parser.parse_args()
pca_mean_vector = np.load(args.mean_shape)
pca_eigen_vectors = np.load(args.eigen_vectors)

output_path = args.out_fold
num_frames = args.num_frames
filename = args.in_file
fs = 8000 # Sampling rate

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# device = torch.device("cpu") 
model = SPCH2FLM().to(device)
model.load_state_dict(torch.load(args.model, map_location=device))

speech, sr = librosa.load(filename)
speech = speech / np.max(np.abs(speech))
speech = librosa.resample(speech, sr, fs)
increment = int(0.04*fs) # Increment rate for 25 FPS videos
lower = 0
predicted = np.zeros((0, pca_mean_vector.shape[1]))
flag = 0
# Pad zeros to start and end of the speech signal
speech = np.insert(speech, 0, np.zeros((int(increment*num_frames/2))))
speech = np.append(speech, np.zeros((int(increment*num_frames/2))))

time_start = clock()

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
    cur_features = torch.from_numpy(cur_features).float().to(device)
    pred = model(cur_features)[0].cpu().data.numpy()
    pred = (pca_mean_vector + np.dot(pred, pca_eigen_vectors))
    predicted = np.append(predicted, np.reshape(pred[0, :], (1, pca_mean_vector.shape[1])), axis=0)
if len(predicted.shape) < 3:
    predicted = np.reshape(predicted, (predicted.shape[0], int(predicted.shape[1]/3), 3))

time_end = clock()

print('tflite run time: ' + str(time_end - time_start) + 'ms')

out_dir = Path(output_path)
filepath, tempfilename = os.path.split(filename)
filename, extension = os.path.splitext(tempfilename)
np.save(out_dir.joinpath(filename+ '.npy'), predicted)


