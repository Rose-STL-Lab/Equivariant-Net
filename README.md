## Paper: 
Rui Wang*, Robin Walters*, Rose Yu [Incorporating Symmetry into Deep Dynamics Models for Improved Generalization](https://arxiv.org/abs/2002.03061), International Conference on Learning Representations 2021

## Abstract:
Recent work has shown deep learning can accelerate the prediction of physical dynamics relative to numerical solvers. However, limited physical accuracy and an inability to generalize under distributional shift limit its applicability to the real world. We propose to improve accuracy and generalization by incorporating symmetries into convolutional neural networks. Specifically, we employ a variety of methods each tailored to enforce a different symmetry. Our models are both theoretically and experimentally robust to distributional shift by symmetry group transformations and enjoy favorable sample complexity. We demonstrate the advantage of our approach on a variety of physical dynamics including Rayleigh–Bénard convection and real-world ocean currents and temperatures. Compare with image or text applications, our work is a significant step towards applying equivariant neural networks to high-dimensional systems with complex dynamics.

## Data Sets
* [Rayleigh–Bénard convection DataSet](https://drive.google.com/drive/folders/1VOtLjfAkCWJePiacoDxC-nrgCREKvrpE?usp=sharing.) 2000 velocity fields (![formula](https://render.githubusercontent.com/render/math?math=2000\times2\times256\times1792))
* [Ocean Current DataSet](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHY_001_024)

## Requirement 
* python 3.6
* pytorch 10.1
* matplotlib
* [e2cnn](https://github.com/QUVA-Lab/e2cnn)

## Description
1. Models/: 
   * Non-Equ/, Magnitude-Equ/, Rotation-Equ/, Uniform-Equ/
     1. model.py: pytorch implementation of models.
     2. utils.py: data loaders, train epoch, validation epoch, test epoch functions.
     3. run_model.py: Scripts to train models.
   * Scale-Equ/
     1. model2d_scalar.py.py: Scale equivariant models for scalar fields.
     2. model_vector.py: Scale equivariant models for vector fields.
     3. train_scalar.py: training functions for scalar fields.
     4. train_vector.py: training functions for vector fields.
     5. run_model.py: Scripts to train models.
     
2. Evaluation/:
   1. Evaluation.py: functions for computing RMSE and Energy Spectrum Errors.
   2. radialProfile.py: FFT functions for calculating energy spectrum.

3. Data Samples/: samples of Rayleigh–Bénard Convection data and Ocean Current data.


## Cite
```
@inproceedings{wang2021incorporating,
title={Incorporating Symmetry into Deep Dynamics Models for Improved Generalization},
author={Rui Wang and Robin Walters and Rose Yu},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=wta_8Hx2KD}
}
```
