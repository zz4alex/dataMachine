# Parkinson-Drawing-1DCNN-BiGRU
Predicting Parkinson’s disease from smartphone finger-drawing trajectories using 1DCNN-BiGRU deep-learning model.

# Overview
This repository contains the full pipeline described in our PLOS ONE manuscript:

Data loader for 9-dimensional touch-screen time series
Windowing & normalization utilities (configurable length/stride)
1DCNN-BiGRU architecture implemented in PyTorch ≥ 2.4
Subject-level, stratified 10-fold cross-validation
Automatic computation of Accuracy, Precision, Recall, F1-score & Specificity
Export of per-participant probabilities to a CSV file
The code has been tested on Windows with NVIDIA GPUs (CUDA 12.1), PyTorch 2.4.0 and scikit-learn 1.4.2.

# Folder Structure

Files:
-- data.zip 
-- 1DCNN-BiGRU-OpenAccess.py
-- README.md
-- LICENSE
