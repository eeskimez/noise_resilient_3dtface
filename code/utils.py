import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from matplotlib import transforms
import argparse, os, fnmatch, shutil
import numpy as np
import math
import copy
import librosa
import subprocess
from tqdm import tqdm

font = {'size'   : 18}
mpl.rc('font', **font)

Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
         [57, 58], [58, 59], [59, 48], [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
         [66, 67], [67, 60]]

Nose = [[27, 28], [28, 29], [29, 30], [30, 31], [30, 35], [31, 32], [32, 33], \
        [33, 34], [34, 35], [27, 31], [27, 35]]

leftBrow = [[17, 18], [18, 19], [19, 20], [20, 21]]
rightBrow = [[22, 23], [23, 24], [24, 25], [25, 26]]

leftEye = [[36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [36, 41]]
rightEye = [[42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [42, 47]]

other = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], \
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], \
         [12, 13], [13, 14], [14, 15], [15, 16]]

faceLmarkLookup = Mouth + Nose + leftBrow + rightBrow + leftEye + rightEye + other

def write_video3D(frames, sound, fs, path, fname, xLim, yLim, zLim, fps, rotate=False):
    try:
        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))
        os.remove(os.path.join(path, fname+'_ws.mp4'))
    except:
        pass

    if len(frames.shape) < 3:
        frames = np.reshape(frames, (frames.shape[0], frames.shape[1]//3, 3))

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    l, = ax.plot3D([], [], [], 'ko', ms=4)

    librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

    if frames.shape[1] == 20:
        lookup = [[x[0] - 48, x[1] - 48] for x in Mouth]
    else:
        lookup = faceLmarkLookup

    lines = [ax.plot([], [], [], 'k', lw=4)[0] for _ in range(3*len(lookup))]

    with writer.saving(fig, os.path.join(path, fname+'.mp4'), 150):
        ax.set_xlim3d(xLim)     
        ax.set_ylim3d(yLim)     
        ax.set_zlim3d(zLim)
        ax.set_xlabel('x', fontsize=48)
        ax.set_ylabel('y', fontsize=48)
        ax.set_zlabel('z', fontsize=48)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        if rotate:
            angles = np.linspace(60, 120, frames.shape[0])
        else:
            angles = np.linspace(60, 60, frames.shape[0])
        for i in tqdm(range(frames.shape[0])):
            ax.view_init(elev=60, azim=angles[i])
            l.set_data(frames[i,:,0], frames[i,:,1])
            l.set_3d_properties(frames[i,:,2])
            cnt = 0
            for refpts in lookup:
                lines[cnt].set_data([frames[i,refpts[1], 0], frames[i,refpts[0], 0]], [frames[i, refpts[1], 1], frames[i,refpts[0], 1]])
                lines[cnt].set_3d_properties([frames[i, refpts[1], 2], frames[i,refpts[0], 2]])
                cnt+=1
            writer.grab_frame()

    cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
    subprocess.call(cmd, shell=True) 

    os.remove(os.path.join(path, fname+'.mp4'))
    os.remove(os.path.join(path, fname+'.wav'))

def getSTFT(speech, sr, winsize, hopsize):
    cnst = 1+(int(int(sr*winsize))/2)
    res_stft =librosa.stft(speech,
                            win_length = int(sr*winsize),
                            hop_length = int(sr*hopsize),
                            n_fft = int(sr*winsize))
    
    stft_mag = np.abs(res_stft)/cnst
    stft_phase = np.angle(res_stft)

    return stft_mag, stft_phase

def main():
    return

if __name__ == "__main__":
    main()