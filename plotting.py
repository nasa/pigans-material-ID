#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import matplotlib.pyplot as plt
import numpy as np


NUM_PREDICT_SAMPLES = 1000


def plot_mean_std_contour(xx, yy, data):

    fig, axes = plt.subplots(*sorted([len(data.keys()), 2])[::-1])
    axes = np.expand_dims(axes, 0) if len(axes.shape) == 1 else axes

    for i, (label, qoi)  in enumerate(data.items()):

        ax = axes[i]

        mean = qoi.mean(axis=0)
        std = qoi.std(axis=0)
    
        cs0 = ax[0].contourf(xx, yy, mean)
        ax[0].set_title(label)
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        cb0 = fig.colorbar(cs0, ax=ax[0])
        cb0.ax.set_ylabel('Mean')
    
        cs1 = ax[1].contourf(xx, yy, std)
        ax[1].set_title(label)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        cb1 = fig.colorbar(cs1, ax=ax[1])
        cb1.ax.set_ylabel('Std')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def plot_samples(xx, yy, outputs, xx_data, yy_data, data):

    fig, axes = plt.subplots(*sorted([len(outputs.keys()), 2])[::-1],
                             figsize=[10,10])
    axes = np.expand_dims(axes, 0) if len(axes.shape) == 1 else axes

    for i, (label, qoi)  in enumerate(outputs.items()):

        qoi_slices = {
            'x': qoi[:, int(qoi.shape[1] / 2), :],
            'y': qoi[:, :, int(qoi.shape[2] / 2)]}
        qoi_pos_slices = {
            'x': xx[int(xx.shape[0] / 2), :],
            'y': yy[:, int(yy.shape[1] / 2)]}

        if label in data.keys():
            data_slices = {
                'y': data[label][:, int(data[label].shape[1] / 2), :],
                'x': data[label][:, :, int(data[label].shape[2] / 2)]}
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
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf


def plot_progress(X_u, batch, trained_model):
    X_u = X_u.numpy()
    batch = batch.numpy()

    x_dim = len(set(X_u[:, 0]))
    y_dim = len(set(X_u[:, 1]))
    xx = X_u[:, 0].reshape(y_dim, x_dim).T
    yy = X_u[:, 1].reshape(y_dim, x_dim).T
    x_bounds = (xx.min(), xx.max())
    y_bounds = (yy.min(), yy.max())

    xx_gen, yy_gen = np.meshgrid(np.linspace(*x_bounds, 100),
                         np.linspace(*y_bounds, 50))
    x = np.c_[xx_gen.ravel(), yy_gen.ravel()].astype('float32')

    results = trained_model.generate(x, NUM_PREDICT_SAMPLES)
    if not isinstance(results, tuple):
        results = (results, ) # only u gen

    bufs = []
    for res, labels in zip(results, (['u1', 'u2'], ['E'])):
        outputs = np.transpose(res.numpy(), (2, 0, 1))
        output_dict = {label: out.reshape(out.shape[0], *xx_gen.shape) \
                    for label, out in zip(labels, outputs)}
        bufs.append(plot_mean_std_contour(xx_gen, yy_gen, output_dict))

        n_sens = xx.shape
        zz = batch.reshape(-1, *n_sens, 2)
        data = {
           'u1': zz[:, :, :, 0],
           'u2': zz[:, :, :, 1]
        }

        bufs.append(plot_samples(xx_gen, yy_gen, output_dict, xx, yy, data))

    return tuple(bufs)
