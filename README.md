## Paper: 
Rui Wang*, Robin Walters*, Rose Yu [Incorporating Symmetry into Deep Dynamics Models for Improved Generalization](https://arxiv.org/abs/2002.03061), International Conference on Learning Representations 2021. 

## Abstract:
Recent work has shown deep learning can accelerate the prediction of physical dynamics relative to numerical solvers. However, limited physical accuracy and an inability to generalize under distributional shift limit its applicability to the real world. We propose to improve accuracy and generalization by incorporating symmetries into convolutional neural networks. Specifically, we employ a variety of methods each tailored to enforce a different symmetry. Our models are both theoretically and experimentally robust to distributional shift by symmetry group transformations and enjoy favorable sample complexity. We demonstrate the advantage of our approach on a variety of physical dynamics including Rayleigh–Bénard convection and real-world ocean currents and temperatures. Compare with image or text applications, our work is a significant step towards applying equivariant neural networks to high-dimensional systems with complex dynamics.

## Data Sets
* [Rayleigh–Bénard convection DataSet](https://roselab1.ucsd.edu/seafile/d/7e7abe7c9c51489daa21/.) 2000 velocity fields (![formula](https://render.githubusercontent.com/render/math?math=2000\times2\times256\times1792))
* [Ocean Current DataSet](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/download)

## Requirement 
* python 3.7
* pytorch >= 1.0.1
* numpy
* scipy
* [e2cnn](https://github.com/QUVA-Lab/e2cnn)

## Description
1. models: pytorch implementation of equivariant models.

2. utils: training functions and dataset classes.
     
3. evaluation: functions for computing RMSE and Energy Spectrum Errors.

4. data_prep.py: preprocess turbulence data and generate training, validation, test and transformed test sets.


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
