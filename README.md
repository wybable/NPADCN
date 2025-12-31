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

<img src="https://raw.github.com/wybable/NPADCN/master/images/Table1.png" width="1024"/><br>

The quantitative statistics of classification results are delineated in **Table I**, where the proposed NPADCN exhibits superior performance across all three public datasets. Using the OA of LK dataset as an illustration, the NPADCN (99.72%) demonstrated notable enhancements compared to the CNN-based CLOLN (99.38%), Attention-based DBDA (99.45%), Transformer-based GLMGT (99.38%), and Deformable Convolution-based DHCN (99.08%) and the recent SClusterFormer (98.31%), showcasing improvements of 0.34%, 0.27%, 0.34%, 0.64%, and 1.41%, respectively. 

<img src="https://raw.github.com/wybable/NPADCN/master/images/Table2.png" width="1024"/>

**Table II** illustrates the accuracy and computational complexity of the ablation models. NPADCN consistently outperforms both the baseline and other attention-guided variants (e.g., DAM and CBAM) across all datasets. Notably, this superior performance is achieved with high efficiency; compared to Baseline-DCN, NPADCN introduces no additional parameters and only a negligible increase in computation time. This demonstrates that NPA-DConv effectively enhances feature extraction without compromising computational efficiency.

<img src="https://raw.github.com/wybable/NPADCN/master/images/Table3.png" width="1024"/>

**Table III** presents the ablation study analyzing the contribution of individual branches. Among dual-branch configurations, the combination of ASpeFE and ASpaFE consistently yields higher accuracy, underscoring the importance of leveraging both spectral and spatial information. Ultimately, the complete NPADCN model integrates all three branches to achieve the highest accuracy across all datasets, confirming the essential contribution of each module to the joint feature extraction.

### Qualitative Results
<img src="https://raw.github.com/wybable/NPADCN/master/images/Fig.png" width="1024"/>

Fig.4 depicts the corresponding classification maps for the LK dataset. It can be seen that the proposed NPADCN model produces the most visually superior classification map with minimal discrete noise, particularly in the edge region.

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

