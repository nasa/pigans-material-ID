#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from pigan.post_processing_scripts.PostProcessor import PostProcessor
from pigan.pigan import PIGAN
from utilities.general import load_train_dataset



#----------Input Parameters---------------#
#File Locatiions
MODEL_DIR = Path(sys.argv[1]) if sys.argv[1] else Path("data/models/")
TRAIN_FILE = Path("data/training_f8000.h5") #relative path to data

#Toggles saving generated data
SAVE_RESULTS = False

#Number of Prediction Samples
NUM_PREDICT_SAMPLES = 1000

#Load Training Data
train_data, boundary_data = load_train_dataset(TRAIN_FILE)

#Load Trained PIGAN Model
trained_model = PIGAN()
trained_model.load(MODEL_DIR)

#Generate Data From Trained Model
NX = 100
NY = 50


def plot_mean_std_contour(xx, yy, data, filesuffix):

    fig, axes = plt.subplots(len(data.keys()), 2)

    for i, (label, qoi)  in enumerate(data.items()):
        mean = qoi.mean(axis=0)
        std = qoi.std(axis=0)
    
        cs0 = axes[i, 0].contourf(xx, yy, mean)
        axes[i, 0].set_title(label)
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        cb0 = fig.colorbar(cs0, ax=axes[i, 0])
        cb0.ax.set_ylabel('Mean')
    
        cs1 = axes[i, 1].contourf(xx, yy, std)
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        cb1 = fig.colorbar(cs1, ax=axes[i, 1])
        cb1.ax.set_ylabel('Std')
    
    plt.tight_layout()
    plt.savefig(f'DEBUG_{filesuffix}.png')


def plot_samples(xx, yy, outputs, xx_data, yy_data, data, filesuffix):
    fig, axes = plt.subplots(len(outputs.keys()), 2, figsize=[10, 10])

    for i, (label, qoi)  in enumerate(outputs.items()):

        qoi_slices = {
            'x': qoi[:, int(qoi.shape[1] / 2), :],
            'y': qoi[:, :, int(qoi.shape[2] / 2)]}
        qoi_pos_slices = {
            'x': xx[int(qoi.shape[1] / 2), :],
            'y': yy[:, int(qoi.shape[2] / 2)]}

        if label in data.keys():
            data_slices = {
                'x': data[label][:, int(data[label].shape[1] / 2), :],
                'y': data[label][:, :, int(data[label].shape[2] / 2)]}
            data_pos_slices = {
                'x': xx_data[int(xx_data.shape[0] / 2), :],
                'y': yy_data[:, int(yy_data.shape[1] / 2)]}

        for ax, axis_label in zip(axes[i, :], ['x', 'y']):
            qoi_slice = qoi_slices[axis_label]
            pos_slice = qoi_pos_slices[axis_label]

            qoi_max = qoi_slice.max(axis=0)
            qoi_min = qoi_slice.min(axis=0)

            ax.fill_between(pos_slice, qoi_min, qoi_max, facecolor='0.7')

            ax.plot(pos_slice, qoi_slice.T, c='0.9', linewidth=0.4)

            if label in data.keys():
                data_slice = data_slices[axis_label]
                data_pos = data_pos_slices[axis_label]

                ax.plot(data_pos, data_slice.T, alpha=0.2)

            ax.set_xlabel(axis_label)
            ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(f'DEBUG_{filesuffix}.png')


if __name__ == '__main__':
    xx_gen, yy_gen = np.meshgrid(np.linspace(0.25, 1.75, 100),
                         np.linspace(0.25, 0.75, 50))
    x = np.c_[xx_gen.ravel(), yy_gen.ravel()].astype('float32')
    outputs = trained_model.generate(x, NUM_PREDICT_SAMPLES, SAVE_RESULTS)
    labels = ['E', 'u1', 'u2']
    output_dict = {label: out.reshape(out.shape[0], *xx_gen.shape) \
                for label, out in zip(labels, outputs)}
    plot_mean_std_contour( xx_gen, yy_gen, output_dict, 'gen_contours')

    x_dim = len(set(train_data['X_u'].numpy()[:, 0]))
    y_dim = len(set(train_data['X_u'].numpy()[:, 1]))
    xx = train_data['X_u'].numpy()[:, 0].reshape(x_dim, y_dim)
    yy = train_data['X_u'].numpy()[:, 1].reshape(x_dim, y_dim)
    n_sens = train_data['Num_U_Sensors']
    data = {
       'u1': train_data['snapshots'].numpy()[:, :n_sens].reshape(-1, *xx.shape),
       'u2': train_data['snapshots'].numpy()[:, n_sens:].reshape(-1, *xx.shape),
    }
    # have to rotate/flip because was lazy when writing swapped components
    xx = np.fliplr(np.rot90(xx, 1))
    yy = np.fliplr(np.rot90(yy, 1))
    data['u1'] = np.flip(np.rot90(data['u1'], 1, axes=(1, 2)), axis=2)
    data['u2'] = np.flip(np.rot90(data['u2'], 1, axes=(1, 2)), axis=2)
    plot_mean_std_contour(xx, yy, data, 'train_contours')

    plot_samples(xx_gen, yy_gen, output_dict, xx, yy, data, 'slices')
