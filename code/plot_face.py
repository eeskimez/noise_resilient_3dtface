import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.lines as mlines
from matplotlib import transforms
from tqdm import tqdm
import os
import subprocess
import librosa

class facePainter():
    inds_mouth = [60, 61, 62, 63, 64, 65, 66, 67, 60]
    inds_top_teeth = [48, 54, 53, 52, 51, 50, 49, 48]
    inds_bottom_teeth = [4, 12, 10, 6, 4]
    inds_skin = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                    57, 58, 59, 48, 49, 50, 51, 52, 52, 53, 54, 55, 56, 57,
                    8, 9, 10, 11, 12, 13, 14, 15, 16,
                    45, 46, 47, 42, 43, 44, 45,
                    16, 71, 70, 69, 68, 0,
                    36, 37, 38, 39, 40, 41, 36, 0]
    inds_lips = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48,
                    60, 67, 66, 65, 64, 63, 62, 61, 60, 48]
    inds_nose = [[27, 28, 29, 30, 31, 27],
                    [30, 31, 32, 33, 34, 35, 30],
                    [27, 28, 29, 30, 35, 27]]
    inds_brows = [[17, 18, 19, 20, 21],
                    [22, 23, 24, 25, 26]]

    def __init__(self, lmarks, speech, fps=25.0, fs=8000):
        lmarks = np.concatenate((lmarks,
                                lmarks[:, [17, 19, 24, 26], :]), 1)[..., :2]
        lmarks[:, -4:, 1] += -0.03
        # lm = lmarks.mean(0)
        # lm = lmarks[600]

        self.lmarks = lmarks
        self.speech = speech
        self.fps = fps
        self.fs = fs

    def plot_face(self, lm):
        plt.axes().set_aspect('equal', 'datalim')

        # make some eyes
        theta = np.linspace(0, 2 * np.pi, 100)
        circle = np.transpose([np.cos(theta), np.sin(theta)])
        for self.inds_eye in [[37, 38, 40, 41], [43, 44, 46, 47]]:
            plt.fill(.013 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .013 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0.5, 0], lw=0)
            plt.fill(.005 * circle[:, 0] + lm[self.inds_eye, 0].mean(),
                    .005 * circle[:, 1] - lm[self.inds_eye, 1].mean(),
                    color=[0, 0, 0], lw=0)
        plt.plot(.01 * circle[:, 0], .01 * circle[:, 1], color=[0, 0.5, 0], lw=0)
        # make the teeth
        # nose bottom to top teeth: 0.037
        # chin bottom to bottom teeth: .088
        plt.fill(lm[self.inds_mouth, 0], -lm[self.inds_mouth, 1], color=[0, 0, 0], lw=0)
        # plt.fill(lm[inds_top_teeth, 0], -lm[inds_top_teeth, 1], color=[1, 1, 0.95], lw=0)
        # plt.fill(lm[inds_bottom_teeth, 0], -lm[inds_bottom_teeth, 1], color=[1, 1, 0.95], lw=0)

        # make the rest
        skin_color = np.array([0.7, 0.5, 0.3])
        plt.fill(lm[self.inds_skin, 0], -lm[self.inds_skin, 1], color=skin_color, lw=0)
        for ii, color_shift in zip(self.inds_nose, [-0.05, -0.1, 0.06]):
            plt.fill(lm[ii, 0], -lm[ii, 1], color=skin_color + color_shift, lw=0)
        plt.fill(lm[self.inds_lips, 0], -lm[self.inds_lips, 1], color=[0.7, 0.3, 0.2], lw=0)

        for ib in self.inds_brows:
            plt.plot(lm[ib, 0], -lm[ib, 1], color=[0.3, 0.2, 0.05], lw=4)

        plt.xlim(-0.15, 0.15)
        plt.ylim(-0.2, 0.18)

    def write_video(self, frames, sound, fs, path, fname, xLim, yLim):
        try:
            os.remove(os.path.join(path, fname+'.mp4'))
            os.remove(os.path.join(path, fname+'.wav'))
            os.remove(os.path.join(path, fname+'_ws.mp4'))
        except:
            pass

        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=self.fps, metadata=metadata)

        fig = plt.figure(figsize=(10, 10))
        # l, = plt.plot([], [], 'ko', ms=4)

        # plt.xlim(xLim)
        # plt.ylim(yLim)

        librosa.output.write_wav(os.path.join(path, fname+'.wav'), sound, fs)

        with writer.saving(fig, os.path.join(path, fname+'.mp4'), 100):
            # plt.gca().invert_yaxis()
            for i in tqdm(range(frames.shape[0])):
                self.plot_face(frames[i, :, :])
                writer.grab_frame()
                plt.clf()
                # plt.close()

        cmd = 'ffmpeg -i '+os.path.join(path, fname)+'.mp4 -i '+os.path.join(path, fname)+'.wav -c:v copy -c:a aac -strict experimental '+os.path.join(path, fname)+'_.mp4'
        subprocess.call(cmd, shell=True) 
        print('Muxing Done')

        os.remove(os.path.join(path, fname+'.mp4'))
        os.remove(os.path.join(path, fname+'.wav'))

    def paintFace(self, path, fname):
        self.write_video(self.lmarks, self.speech, self.fs, path, fname, [-0.15, 0.15], [-0.2, 0.18])
