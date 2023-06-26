#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

from plotting import *
from pathlib import Path
from pigan.post_processing_scripts.PostProcessor import PostProcessor
from pigan.pigan import PIGAN
from utilities.general import load_train_dataset



#----------Input Parameters---------------#
#File Locatiions
nargs = len(sys.argv)
MODEL_DIR = Path(sys.argv[1] if nargs >= 2 else "data/models/")
TRAIN_FILE = Path(sys.argv[2] if nargs >= 3 else "data/training_f8000.h5")

#Toggles saving generated data
SAVE_RESULTS = False

#Number of Prediction Samples
NUM_PREDICT_SAMPLES = 1000

#Load Training Data
train_data, boundary_data = load_train_dataset(TRAIN_FILE)

#Generate Data From Trained Model
NX = 100
NY = 50


for model_dir in set([f.parent for f in MODEL_DIR.glob('*.h5')]):

    trained_model = PIGAN()
    trained_model.load(model_dir)

    x_dim = len(set(train_data['X_u'].numpy()[:, 0]))
    y_dim = len(set(train_data['X_u'].numpy()[:, 1]))
    xx = train_data['X_u'].numpy()[:, 0].reshape(x_dim, y_dim)
    yy = train_data['X_u'].numpy()[:, 1].reshape(x_dim, y_dim)
    x_bounds = (xx.min(), xx.max())
    y_bounds = (yy.min(), yy.max())

    xx_gen, yy_gen = np.meshgrid(np.linspace(*x_bounds, 100),
                         np.linspace(*y_bounds, 50))
    x = np.c_[xx_gen.ravel(), yy_gen.ravel()].astype('float32')
    outputs = trained_model.generate(x, NUM_PREDICT_SAMPLES, SAVE_RESULTS)
    labels = ['u1', 'u2']
    output_dict = {label: out.reshape(out.shape[0], *xx_gen.shape) \
                for label, out in zip(labels, outputs)}
    plot_mean_std_contour(xx_gen, yy_gen, output_dict)
    plt.savefig(model_dir / 'gen_contours.png')
    plt.close('all')

    n_sens = train_data['Num_U_Sensors']
    xx = train_data['X_u'].numpy()[:, 0].reshape(*n_sens)
    yy = train_data['X_u'].numpy()[:, 1].reshape(*n_sens)
    zz = train_data['snapshots'].numpy().reshape(-1, *n_sens, 2)
    data = {
       #'u1':
       #train_data['snapshots'].numpy()[:, :n_sens].reshape(-1, *xx.shape),
       #'u2':
       #train_data['snapshots'].numpy()[:, n_sens:].reshape(-1, *xx.shape),
        'u1': zz[:, :, :, 0],
        'u2': zz[:, :, :, 1]
    }
    # have to rotate/flip because was lazy when writing swapped components
    #xx = np.fliplr(np.rot90(xx, 1))
    #yy = np.fliplr(np.rot90(yy, 1))
    #data['u1'] = np.flip(np.rot90(data['u1'], 1, axes=(1, 2)), axis=2)
    #data['u2'] = np.flip(np.rot90(data['u2'], 1, axes=(1, 2)), axis=2)
    plot_mean_std_contour(xx, yy, data)
    plt.savefig(model_dir / 'train_contours.png')
    plt.close('all')

    plot_samples(xx_gen, yy_gen, output_dict, xx, yy, data)
    plt.savefig(model_dir / 'slices.png')
    plt.close('all')
