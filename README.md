# NPADCN
This repo is an implementation of Non-Parameter Attention Guided Deformable Convolution Network(NPADCN) for Hyperspectral Image Classification.

## Overview

### Framework of NPADCN
<img src="https://raw.github.com/wybable/NPADCN/master/images/NPADCN Framework.png" width="1024"/><br>

### Architecture of Non-Parameter Attention(NPA)
<img src="https://raw.github.com/wybable/NPADCN/master/images/NPA.png" width="1024"/><br>

### Architecture of Non-Parameter Attention Guided Deformable Convolution(NPA-DConv)
<img src="https://raw.github.com/wybable/NPADCN/master/images/NPA-DConv.png" width="1024"/><br>

## Requirements
- Python 3
- pytorch (1.0.0), torchvision (0.2.2) / pytorch (1.2.0), torchvision (0.4.0) 
- numpy, PIL
- Visual Studio 2015

## Build [Deformable 3D Convolution]
***Compile deformable convolution***: <br>
1. Cd to ```code/dcn```.
2. For Windows users, run  ```cmd make.bat```. For Linux users, run ```bash make.sh```. The scripts will build 3D deformable convolution automatically and create some folders.
3. We offer customized settings for 3d dimension (e.g., Bands, Height, Width). See ```code/dcn/test.py``` for more details.

## Datasets

Download the **WHU-Hi-LongKou** (LK) dataset(550 × 400, 270 bands, 9 classes) and **Pavia University** (PU) dataset(610 × 340, 103 bands, 9 classes) in https://pan.baidu.com/s/15or9q9qhJkOLvkd4M4Pk0w?pwd=smi9 (Code: smi9) and extract the datasets to `code/data`.

## Results

### Quantitative Results

<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/table1.JPG" width="1024" />


<img src="https://raw.github.com/XinyiYing/D3Dnet/master/images/table2.JPG" width="550"/>

We have organized the Matlab code framework of Video Quality Assessment metric T-MOVIE and MOVIE. [<a href="https://github.com/XinyiYing/MOVIE">Code</a>] <br> Welcome to have a look and use our code.

### Qualitative Results
<img src=https://raw.github.com/XinyiYing/D3Dnet/master/images/compare.jpg>

A demo video is available at https://wyqdatabase.s3-us-west-1.amazonaws.com/D3Dnet.mp4

## Citiation
```
@article{NPADCN,
  author = {Wang, Yibo and Zhang, Xia and Qi, Wenchao and Wang, Jinnian and Zhou, Zhi and Yang, Yingpin},
  title = {Non-Parameter Attention Guided Deformable Convolution Network for Hyperspectral Image Classification},
  journal = {IEEE Geoscience and Remote Sensing Letters},
}
```

## Acknowledgement
This code is built on [[D3Dnet]](https://github.com/XinyiYing/D3Dnet). We thank the authors for sharing their codes.

## Contact
Please contact us at ***wangyb@gzhu.edu.cn*** for any question.

