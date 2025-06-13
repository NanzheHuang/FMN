# Learning Feature-enhanced Multi-scale Network for SE(3)-Equivariant Motion Prediction


In this work, we propose the Feature-enhanced Multi-scale Network (FMN) to predict SE(3)-equivariant motion in 3D multi-body systems. FMN uses a frame-transform strategy to ensure strict equivariance and includes three feature-enhancement mechanisms: augmenting initial data with spectral and vectorized features, using a multi-scale graph strategy to capture spatial information, and integrating memory to incorporate historical knowledge. We also develop a directional fusion approach to update particle positions. Experiments show that these innovations improve prediction accuracy.


## Overview

<img src='images/pipeline00.png'> 


## Dependencies
```
python==3.8
torch==2.1.2
torch-geometric==2.2.0
torch-scatter==2.1.2+pt21cu121
torch-sparse==0.6.18+pt21cu121
scikit-learn==1.3.2
```

## Data Preparation

+ MD17 & MD22: Download [MD17, MD22 in npz files](http://quantum-machine.org/datasets/) and save in `data/md17/`, `data/md22`.
+ Mocap: The processed data has been placed in the `data/mocap/`.

## Usage

### Prediction task on MD17
Training & Testing
```
python main_md17.py 
```


### Prediction task on MD22
Training & Testing
```
python main_md22.py 
```


### Prediction task on Mocap
Training & Testing
```
python main_mocap.py 
```
The results of training, validation, and testing will be saved in the `res/` folder.
