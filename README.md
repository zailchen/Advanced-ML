# STAT 5242 Advanced Machine Learning Final Project
## Columbia University Fall 2018


This repository contains python codes for the final AML project.

## Dataset
Data we used for this project is available at https://paris-saclay-cds.github.io/autism_challenge/ (only `public set` is available) and initially published for competition: Imaging-psychiatry challenge: predicting autism (IMPAC). 

## Environment

See `Requirement.txt` for modules we use

## Usage

#### Preprocessing

- Feature Extractor

`feature_extractor.py` - extract correlation matrices for each fmri feature

`FeatureExtractor_dtw.py` - extract dtw distance for each fmri feature (not in final use)

- Data Split

`split_train_test.py` - randomly split training and test dataset, and fix for model input

#### Models

- Logistic Regression

`simple_LR.py` - simple logistic regression classifier

- AutoEncoder (dependency: `model.py`, `utils.py`, ref: https://github.com/lsa-pucrs/acerta-abide)

`autoencoder_ensemble.py` - ensemble AutoEncoder classifier

`autoencoder_basc064.py` - AutoEncoder classifier using only `basc064` feature (not in final use)

`autoencoder_all.py` - AutoEncoder classsifier using all features as one time input (not in final use)

- CNN

`CNN_ensemble.py` - ensemble CNN classifier (ref: https://github.com/MRegina/connectome_conv_net)

`CNN_original_trial.py` - CNN classifier with square filters and simply network (not in final use)

`draw_cnn_our_version.py` - code to draw cnn (ref: https://github.com/gwding/draw_convnet)


## Contributors
- Ziyu Chen ([`zc2393`])
- Jongwoo Choi ([`jc4816`])
- Qianqian Hu ([`qh2185`])


