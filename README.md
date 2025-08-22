## EEG Motor Imagery Project

This project processes EEG data (BCI Competition IV, 2a) for motor imagery classification.
It follows a clean, modular pipeline:

Exploration → inspect raw EEG & metadata

Preprocessing → clean with ICA, filtering, re-referencing

Epoching → cut into trials, reject artifacts, balance classes

Features & ML → extract features and train classifiers

## Project Structure

project-root/
│
├── data/
│   ├── raw/          <- original .gdf EEG files (from BCI Competition IV 2a)
│   ├── processed/    <- cleaned .fif files (after ICA artifact removal)
│   └── features/     <- NumPy arrays (X, y) ready for ML
│
├── notebooks/
│   ├── 01_exploration.ipynb   <- load raw data, inspect channels/events
│   ├── 02_epoching.ipynb      <- epoch trials, reject artifacts, balance dataset
│   └── 03_features.ipynb      <- feature extraction + machine learning (next phase)
│
├── .gitignore
└── README.md

## Pipeline Overview
Phase 1: Exploration (01_exploration.ipynb)

    Load raw .gdf EEG files

    Inspect channel names, sampling frequency

    Set channel types (EEG/EOG)

    Apply montage (10–20 system)

    Filter and clean with ICA (remove eye artifacts)

    Save cleaned data to data/processed/

Phase 2: Epoching (02_epoching.ipynb)

    Load cleaned EEG (.fif)

    Extract events from annotations

    Keep only motor imagery classes (left, right, foot, tongue → codes 769–772)

    Epoch trials (0–4s after cue)

    Automatic artifact rejection (reject EEG >150 µV, EOG >250 µV)

    Balance classes (undersample to smallest class)

    Save final balanced dataset (X.npy, y.npy) to data/features/

Phase 3: Features + Machine Learning (03_features.ipynb)

    (to be developed)

    Extract features (PSD, CSP, bandpower, etc.)

    Train classifiers (LDA, SVM, Logistic Regression)

    Evaluate with cross-validation

    Compare models & report accuracy

## Requirements

Python 3.10+

Libraries:

    pip install mne numpy matplotlib scikit-learn

## Key Notes

All raw data (.gdf) must be placed in data/raw/.

Cleaned/preprocessed .fif files go in data/processed/.

Features for ML go in data/features/.

Large binary files (.fif, .npy) should be added to .gitignore if using GitHub.

## Next Steps

Implement feature extraction (CSP, PSD, bandpower).

Train & evaluate classifiers (LDA, SVM).

Extend pipeline to multiple subjects.

Package results for reproducibility.