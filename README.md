# Unsupervised machine learning of topological phase transitionsfrom experimental data
This repository contains the notebooks and scripts to reconstruct the results in our paper https://arxiv.org/abs/2101.05712.

# Where to get the data
We provide access to the data via zenodo 10.5281/zenodo.4459311. You can download the zip archieve there and unzip it. It will contain the data.

# Structure
**.**: The root folder contains the notebooks for the different figures. In general they can be run individually. If you want to use them to experiment or for your own data you need to run them according to their order in the paper.

**data**: The data folder contains the data. This folder needs to be extracted from the zip archive we provide via zenodo 10.5281/zenodo.4459311. You'll find 7 hdf5 files there. They contain the image pixel data and the experimental parameters. See the notebooks for details. We provide the raw dataset cropped to 56x56 pixels which restricts us to data of the first BZ. The files with the rephased postfix correspond to the data which is rephased to a fixed micromotion phase by our VAE.
* phase_diagram_56.h5 The original data of the complete phase diagram cropped to 56x56px.
* phase_diagram_rephased.h5 The rephased data to a micromotion phase of 0 of the phase_diagram_56.h5 data.
* phase_diagram_theory.h5 The theory prediction for the complete Haldane phase diagram
* single_cut_56.h5 The original data of a single cut fur the fixed shaking phase of -90°
* single_cut_rephased.h5 The rephased data to a micromotion phase of 0 of the single_cut_56.h5 data.
* validation_single_cut_56.h5 A validation data set for a shaking phase of -90°. We use it to validate the micromotion removal with influence functions.
* validation_single_cut_rephased.h5  The rephased data to a micromotion phase of 0 of the validation_single_cut_data_56.h5 data.


**networks**: We provide our trained networks here.

**lib**: We save our helper files there.
