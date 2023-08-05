## Paper: 
Rui Wang*, Robin Walters*, Rose Yu [Incorporating Symmetry into Deep Dynamics Models for Improved Generalization](https://arxiv.org/abs/2002.03061), International Conference on Learning Representations 2021. (*equal contribution)

## Abstract:
Recent work has shown deep learning can accelerate the prediction of physical dynamics relative to numerical solvers. However, limited physical accuracy and an inability to generalize under distributional shift limit its applicability to the real world. We propose to improve accuracy and generalization by incorporating symmetries into convolutional neural networks. Specifically, we employ a variety of methods each tailored to enforce a different symmetry. Our models are both theoretically and experimentally robust to distributional shift by symmetry group transformations and enjoy favorable sample complexity. We demonstrate the advantage of our approach on a variety of physical dynamics including Rayleigh–Bénard convection and real-world ocean currents and temperatures. Compare with image or text applications, our work is a significant step towards applying equivariant neural networks to high-dimensional systems with complex dynamics.

## Data Sets
* [Rayleigh–Bénard convection DataSet](https://roselab1.ucsd.edu/seafile/d/7e7abe7c9c51489daa21/.) 2000 velocity fields (![formula](https://render.githubusercontent.com/render/math?math=2000\times2\times256\times1792))
* [Ocean Current DataSet](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHY_001_024)

## Requirements
- To install requirements
```
pip install -r requirements.txt
```

## Description
1. models: pytorch implementation of equivariant models.

2. utils.py: training functions and dataset classes.
     
3. evaluation: code for computing Energy Spectrum Errors.

4. data_prep.py: preprocess RBC and Ocean data.

## Instructions
### Dataset and Preprocessing
- Download [RBC data and Ocean Data.](https://roselab1.ucsd.edu/seafile/d/7e7abe7c9c51489daa21/.) and put 'rbc_data.pt' and all the ocean NetCDF files in the same directory as data_prep.py. Due to the unavailability of the ocean data we previously downloaded from [Copernicus](https://resources.marine.copernicus.eu/?option=com_csw&view=details&product_id=GLOBAL_ANALYSIS_FORECAST_PHY_001_024) for the years 2016 to 2017, we conducted the experiments again using data from 2021 to 2022 with the same latitude and longitude range.

-  run data_prep.py to preprocess RBC and Ocean data and generate training, validation, test (and transformed test sets).
```
python data_prep.py
```

### Training
- Train Equiv and Non-Equiv ResNets and Unets on RBC data and Ocean data.
```
sh run_rbc.sh
```

### New results on ocean currents data from 2021 to 2022 

|           |      RMSE     |               |      ESE      |               |
|-----------|:-------------:|:-------------:|:-------------:|:-------------:|
|           |   Test_time   |  Test_domain  |   Test_time   |  Test_domain  |
| ResNet    |   0.82-0.02   |   1.42-0.02   |   1.09-0.13   |   1.23-0.14   |
| Equ_UM    |   0.80-0.01   |   1.29-0.01   |   1.10-0.02   |   1.19-0.03   |
| Equ_Mag   |   0.79-0.01   |   1.29-0.00   |   1.01-0.01   |   1.02-0.03   |
| Equ_Rot   |   0.78-0.01   |   1.26-0.01   | **0.88-0.01** | **0.98-0.01** |
| Equ_Scale | **0.76-0.03** |   **1.25-0.02**   |   0.98-0.04   |   1.00-0.03   |
| Unet      |   0.78-0.03   |   1.30-0.02   |   1.17-0.08   |   1.25-0.05   |
| Equ_UM    |   0.75-0.01   |   1.32-0.02   |   1.19-0.02   |   1.29-0.00   |
| Equ_Mag   | **0.74-0.01** |   1.27-0.02   |   0.94-0.01   |   1.00-0.03   |
| Equ_Rot   |   0.74-0.02   | **1.08-0.02** | **0.63-0.00** | **0.85-0.01** |
| Equ_Scale |   0.75-0.00   |   1.12-0.00   |   0.89-0.04   |   0.98-0.02   |


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
