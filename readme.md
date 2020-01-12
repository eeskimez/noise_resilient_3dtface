# Noise-Resilient Training Method For Face Landmark Generation From Speech

The code for the following [paper](https://ieeexplore.ieee.org/document/8871109).

You can find the project page [here](http://www2.ece.rochester.edu/projects/air/projects/3Dtalkingface.html).

## Installation

#### Install the required Python packages:
```
pip install -r requirements.txt
```

#### It also depends on the following packages:
* ffmpeg --- 3.4.1

The code has been tested on Ubuntu 16.04 and OS X Sierra and High Sierra. 

* Note that the new versions of the PyTorch will yield different results due to numerical differences.

## Code Example

The generation code has the following arguments:

* -i --- Input folder containing speech files
    * See [this](http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load) link for supported audio formats.
* -m --- Input pre-trained talking face landmarks model 
* -o --- Output path

You can run the following code to test the system:

```
python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../results/1D_CNN/
```

```
python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN_NR.pt -o ../results/1D_CNN_NR/
```

```
python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN_TC.pt -o ../results/1D_CNN_TC/ --temporal_condition
```
## Face landmarks Extraction from Videos

Please see the following links for extracting face landmarks:

[3D face landmarks](https://www.adrianbulat.com/face-alignment)

[DLIB](http://dlib.net/)

For face normalization, please refer to [this repo.](https://github.com/lelechen63/ATVGnet)

## Training

The training code has the following arguments:

* -i --- Input hdf5 file containing training data
* -n --- Input hdf5 file containing noise files (optional)
* --temporal_condition --- boolen value to enable autoregression model (optional)
* -o --- Output folder path to save the model

Usage:

Base model training:

```
python train.py -i path-to-hdf5-train-file/ -o output-folder-to-save-model-file
```

Base model noise_resilient training:

```
python train.py -i path-to-hdf5-train-file/ -n path-to-hdf5-noise-file/ -o output-folder-to-save-model-file
```

Autoregressive model training:

```
python train.py -i path-to-hdf5-train-file/ --temporal_condition -o output-folder-to-save-model-file
```

Autoregressive noise-resilient model training:

```
python train.py -i path-to-hdf5-train-file/ -n path-to-hdf5-noise-file/ --temporal_condition -o output-folder-to-save-model-file
```

## Citation

```
@ARTICLE{seeskimeznr3dtface,
         author={S. E. {Eskimez} and R. K. {Maddox} and C. {Xu} and Z. {Duan}},
         journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
         title={Noise-Resilient Training Method for Face Landmark Generation From Speech},
         year={2020},
         volume={28},
         pages={27-38},
         doi={10.1109/TASLP.2019.2947741},
         ISSN={2329-9304}
}
```
