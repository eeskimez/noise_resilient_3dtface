# Noise-Resilient Training Method For Face Landmark Generation From Speech

You can find the project page [here](http://www2.ece.rochester.edu/projects/air/projects/3Dtalkingface.html).

## Installation

#### Install the required Python packages:
```
pip install -r requirements.txt
```

#### It also depends on the following packages:
* ffmpeg --- 3.4.1

The code has been tested on Ubuntu 16.04 and OS X Sierra and High Sierra. 

## Code Example

The generation code has the following arguments:

* -i --- Input folder containing speech files
    * See [this](http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load) link for supported audio formats.
* -m --- Input pre-trained talking face landmarks model 
* -o --- Output path
* `-s --save_prediction` save the predicted landmarks and speech array in the folder specified by the `-o` option and disable generation of animation
* `-l --load_prediction` load predictions from the folder specified by the `-i` option and generate a painted face animation in the folder specified by the `-o` option. This option expects the input folder to contain pairs of files with the same name but different extensions - `.wav` and `.npy`

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
Save the landmarks predicted and speech vector using the [ID_CNN](pre_trained/1D_CNN.pt) model from audio in `../speech_samples/` to an [NPY format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) file in `replic/pred_out/`

    python generate.py -i ../speech_samples/ -m ../pre_trained/1D_CNN.pt -o ../replic/pred_out/ -s  

Load landmarks from external files in `replic/samples/identity_removed/` and generate animation in `replic/anim_out/`

    python generate.py -i ../replic/samples/identity_removed/ -m ../pre_trained/1D_CNN.pt -o ../replic/anim_out/ -l

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
