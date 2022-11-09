# HyperRes: Efficient Hypernetwork-Based Continuous Image Restoratio
by [Shai Aharon](https://www.linkedin.com/in/shai-aharon-b4495273/) and [Dr. Gil Ben-Artzi](http://www.gil-ba.com/)

This is the official implementation of _HyperRes: Efficient Hypernetwork-Based Continuous Image Restoration_ as described in 
[Link: Arxiv](https://arxiv.org/).

![](figures/train_arc.png "HyperRes: Efficient Hypernetwork-Based Continuous Image Restoration")

We present a hypernetwork framework which, for a given corruption level, dynamically generates weights for a 
main network which is optimized to restore the given corruption level.

<p align="center">
<img src="figures/comapr_sr_noise.png" width="800"/>
</p>

For evaluation, we compared our method to 18 different networks, each has been separately 
trained on a single noise level from 5 to 90. Those networks determine the maximum accuracy that a network with similar 
architecture could achieve.
Our method surpasses _SOTA_ results, and achieves a near optimal accuracy on all the range, evan on extrapolation.

The graph below, compares our method and STOA methods to the optimal accuracy (the zero line).

<p align="center">
<img src="figures/noise_diff_compare.png" alt="Evaluation comparison on different noise levels" width="600"/>
</p>

[//]: # (![]&#40;figures/sr.gif&#41;)

## Requirements

### Dependency

Our code is based on the PyTorch library
* PyTorch 1.5.11+

Run the following command line to install all the dependencies


    pip install -r requirements.txt

### Dataset

We used the DIV2K dataset for training all tasks. For evaluation, we used 
a different dataset for each task, in compliance to the common benchmarks, as follows.

| Task              | Dataset   | Comments                                              |
| ----------------- | --------- | ----------------------------------------------------- |
| Training			| DIV2K		| Used to train all tasks								|
| DeNoising         | BDSD68    |                                                       |
| DeJPEG            | LIVE1     | Trained and tested on Luminance (grayscale)           |
| Super-Resolution  | Set5      | Trained on RGB and tested on Luminance (grayscale)    |

Links:
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [CBSD68] (https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
- [LIVE1](https://live.ece.utexas.edu/research/quality/subjective.htm)
- [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html)

## Pre-trained Models

### We supply pre-trained models for each task

| Task              | Levels                | Link                                      |
| -------------     | --------------------- | ----------------------------------------- |
| DeNoising         | [15,35,50,75]         | [Link](pre_trained/Noise_4/latest.pth)    |
| DeNoising         | [5,25,45,65,85]       | [Link](pre_trained/Noise_5/latest.pth)    |

## Training
### Dataset File structure
To train/test the dataset should be defined as follows:

```
[main folder]─┐
              ├─► train ──┐
              │           ├─► n_10 ──┐
              ├─► valid   │          ├─► *.png
              │           ├─► n_15   │
              └─► test    │          ├─► *.png
                          .          │
                          .          │
                          .          │
                          .          │
                          .          └─► *.png
                          │
                          ├─► n_90
                          │
                          └─► clean

```

- The structure for the valid/test folder is the same as train.
- 'n' is for DeNoising, 'j' is used for DeJPEG and 'sr' for Super Resolution
- [n/j/sr]_[Number] is the corruption folders, the letters represent the task, and the number is the corruption level.
- The images name in the 'clean' folder should mirror the names in the corruption folders


Train command:
```shell
python train_main.py       \
        --data [path to data]   \
        --data_type [n,sr,j]    \
        --lvls 15 35 50 75      \
        --checkpoint [SavePath] \
        --device [cpu/cuda]     \       
        --no_bn                 
```
## Testing

```shell
python test_main.py                          \
        --data [path to data]                     \ 
        --data_type [n,sr,j]                      \
        --lvls [15 45 75]                         \
        --valid [test folder inside data]         \                              
        --weights [path to weights file (*.pth)]  \ 
        --device [cpu/cuda]                       \
        --no_bn
```

## Live Demo
```shell
python live_demo.py                                       \
        --input [Path to image folder or image]           \
        --data_type [n,sr,j]                              \
        --checkpoint [Path to weights file (*.pth)]       \ 
        --no_bn 
```

## Cite

[HyperRes paper](https://arxiv.org/abs/2206.05970):

```
@misc{https://doi.org/10.48550/arxiv.2206.05970,
  doi = {10.48550/ARXIV.2206.05970},  
  url = {https://arxiv.org/abs/2206.05970},  
  author = {Aharon, Shai and Ben-Artzi, Gil},  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},  
  title = {One Size Fits All: Hypernetwork for Tunable Image Restoration},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
