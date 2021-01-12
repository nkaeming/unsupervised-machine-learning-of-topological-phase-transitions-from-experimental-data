# Unsupervised machine learning of topological phase transitionsfrom experimental data
This repository contains the notebooks and scripts to reconstruct the results in this paper [insert arxiv link].

# Where to get the data
We provide access to the data via zenodo [insert doi]. You can download the zip archieve there and unzip it. It will contain the data as well as the trained models.

# Structure
**.**: The root folder contains the notebooks for the different figures. In general they can be run individually. If you want to use them to experiment or for your own data you need to run them according to their order in the paper.

**data**: The data folder contains the data. This folder needs to be extracted from the zip archive we provide via zenodo [insert doi]. You'll find 7 hdf5 files there. They contain the image pixel data and the experimental parameters. The the notebooks for details. We provide the raw dataset cropped to 56x56 pixels which restricts us to data for the first BZ. The files with the rephased postfix correspond to the data which is rephased to a fixed micromotion phase by our VAE.

**trained_networks**: We rpovide our trained networks here.

# How to reconstruct the plots
