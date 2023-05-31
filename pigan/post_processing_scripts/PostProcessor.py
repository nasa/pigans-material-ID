#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:03:42 2020

@author: tlewitt
"""
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.stats import gaussian_kde

class PostProcessor():
    def __init__(self, x_points, E_true, E_gen, ux_true, ux_gen, uy_true, uy_gen):

        self.X = x_points
        self.num_sensors = self.infer_test_points()
        
        self.E_true = E_true
        self.ux_true = ux_true
        self.uy_true = uy_true
        
        self.E_gen = E_gen
        self.ux_gen = ux_gen
        self.uy_gen = uy_gen

        self.save_dir = Path("data/plots/")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def infer_test_points(self):

        length = self.X.shape[0]
        num_sensors_x = np.argmax(self.X[:, 0]) + 1
        num_sensors_y = int(length / num_sensors_x)
        return (num_sensors_x, num_sensors_y)
        
    def create_paper_graphs(self):
        '''
        Plots figures 2-5 from the paper
        '''
             
        true_samples_save_name = "true_sample_contours.pdf"
        self.plot_true_samples(true_samples_save_name)
         
        generated_samples_save_name = "generated_sample_contours.pdf"
        self.plot_generated_samples(generated_samples_save_name)
        
        mean_diff_save_name = "E_mean_diff_contour.pdf"
        std_diff_save_name = "E_std_diff_contour.pdf"
        self.plot_mean_std_diff_field(mean_diff_save_name, std_diff_save_name)

        pointwise_pdf_save_name = "E_pointwise_prob_density.pdf"
        self.plot_pointwise_PDF(pointwise_pdf_save_name)
        
        correlation_comp_save_name = "E_correlation_comparison.pdf"
        self.plot_correlation(correlation_comp_save_name)



    def plot_true_samples(self, save_name):
        '''
        Figure Two from the paper
        '''
        xmesh, ymesh = self._get_grid(self.X, self.num_sensors)
        
        #Draw three random samples of each:
       

        sample_indices = np.array(random.sample(range(1, self.ux_gen.shape[0]), 3))
        
        E_samples = self.E_true[sample_indices, :]
        ux_samples = self.ux_true[sample_indices, :]
        uy_samples = self.uy_true[sample_indices, :]
        
        self._plot_sample_contours(xmesh, ymesh, E_samples, ux_samples,
                                   uy_samples, save_name)    

    def plot_generated_samples(self, save_name):
        '''
        Figure Three from the paper
        '''
        xmesh, ymesh = self._get_grid(self.X, self.num_sensors)

        #Draw three random samples of each:
        sample_indices = np.array(random.sample(range(1, self.ux_gen.shape[0]), 3))
        
        E_samples_gen = self.E_gen[sample_indices, :]
        ux_samples_gen = self.ux_gen[sample_indices, :]
        uy_samples_gen = self.uy_gen[sample_indices, :]
        self._plot_sample_contours(xmesh, ymesh, E_samples_gen, ux_samples_gen, 
                                   uy_samples_gen, save_name)

    def plot_mean_std_diff_field(self, save_mean_name, save_std_name):
        '''
        Figure Four from the paper
        '''
        xmesh, ymesh = self._get_grid(self.X, self.num_sensors)
        
        E_mean_gen = np.mean(self.E_gen, axis=0).reshape((self.num_sensors[0], 
                            self.num_sensors[1]))
        E_mean_true = np.mean(self.E_true, axis=0).reshape((self.num_sensors[0], 
                             self.num_sensors[1]))

        E_std_gen = np.std(self.E_gen, axis=0).reshape((self.num_sensors[0], 
                          self.num_sensors[1]))
        E_std_true = np.std(self.E_true, axis=0).reshape((self.num_sensors[0], 
                           self.num_sensors[1]))

        E_mean_diff = E_mean_true - E_mean_gen
        E_std_diff = E_std_true - E_std_gen

        self._plot_contour_meanstddiff(xmesh, ymesh, E_mean_diff, save_mean_name)
        self._plot_contour_meanstddiff(xmesh, ymesh, E_std_diff, save_std_name)
    
    def plot_pointwise_PDF(self, save_name):
        '''
        Figure Five part A from the paper
        '''
        xmin = 0.5
        xmax = 2
        a_coord = np.array([0.75, 0.75])
        b_coord = np.array([0.5, 0.5])
        c_coord = np.array([0.25, 0.25])
         
        #Reference PDF should theorectically be the same every, grab at (0.5, 0.5)
        ref_pdf = self._get_kde_pdf_at_coord(self.X, self.E_true, b_coord)
        
        a_pdf = self._get_kde_pdf_at_coord(self.X, self.E_gen, a_coord)
        b_pdf = self._get_kde_pdf_at_coord(self.X, self.E_gen, b_coord)
        c_pdf = self._get_kde_pdf_at_coord(self.X, self.E_gen, c_coord)
        
        axis_font = {'fontname':'Arial', 'size':20, 'weight':'normal'}
        labelSize = 20
        legSize = 18
        linewidth = 2
        x_plot = np.linspace(xmin, xmax, 1000)

        plt.figure()
        ax = plt.subplot()
        plt.plot(x_plot, ref_pdf(x_plot), 'k-', linewidth=linewidth, label="Reference")
        plt.plot(x_plot, a_pdf(x_plot), 'r--', linewidth=linewidth, 
                 label="Generated (a)")
        plt.plot(x_plot, b_pdf(x_plot), 'm-.', linewidth=linewidth,
                 label="Generated (b)")
        plt.plot(x_plot, c_pdf(x_plot), 'g:', linewidth=linewidth, 
                 label="Generated (c)")
        plt.xlabel("E", **axis_font)
        plt.ylabel("p(E)", **axis_font)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)
 
        plt.legend(prop={"size":legSize})
        plt.tight_layout()
        plt.savefig(self.save_dir.joinpath(save_name))
        plt.show()

    def plot_correlation(self, save_name):
        '''
        Figure Five part B from the paper
        '''
        pts_of_interest = [0.25, 0.5, 0.75]
        
        ref_corr = self._get_corr_at_y_coord(self.X, self.E_true, pts_of_interest[1])
        a_corr = self._get_corr_at_y_coord(self.X, self.E_gen, pts_of_interest[2])
        b_corr = self._get_corr_at_y_coord(self.X, self.E_gen, pts_of_interest[1])
        c_corr = self._get_corr_at_y_coord(self.X, self.E_gen, pts_of_interest[0])
        
        axis_font = {'fontname':'Arial', 'size':20, 'weight':'normal'}
        labelSize = 20
        legSize = 18
        linewidth = 2
        
        x_plot = np.linspace(0, 1, self.num_sensors[0])
        
        plt.figure()
        ax = plt.subplot()
        plt.plot(x_plot, ref_corr, 'k-', linewidth=linewidth, label="Reference")
        plt.plot(x_plot, a_corr, 'r--', linewidth=linewidth,
                 label="Generated (A-A)")
        plt.plot(x_plot, b_corr, 'm-.', linewidth=linewidth,
                 label="Generated (B-B)")
        plt.plot(x_plot, c_corr, 'g:', linewidth=linewidth,
                 label="Generated (C-C)")
        plt.xlabel("x", **axis_font)
        plt.ylabel(r'$C_{\bar{y}}(x, 0.5)$', **axis_font)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)
        
        plt.legend(prop={"size":legSize})
        plt.tight_layout()
        plt.savefig(self.save_dir.joinpath(save_name))
        plt.show()

    def _get_kde_pdf_at_coord(self, coord_data, value_data, pdfcoord):

        plot_index = np.argmin(np.linalg.norm(coord_data- pdfcoord, axis=1))
        data_at_coord = value_data[:, plot_index]
        pdf_at_coord = gaussian_kde(data_at_coord)
        return pdf_at_coord

    def _get_corr_at_y_coord(self, x_data, value_data, corr_y_coord):

        y_index = np.argmin(np.abs(x_data[:, 1] - corr_y_coord))
        y_indices = np.arange(y_index, y_index+self.num_sensors[0])

        value_data_along_y = value_data[:, y_indices]
        mid_index = int(np.floor(len(y_indices) / 2))
        corr_matrix = np.corrcoef(value_data_along_y.T)

        return corr_matrix[:, mid_index]

    def _get_grid(self, x_gen, num_sensors):

        w = np.max(x_gen[:, 0])
        L = np.max(x_gen[:, 1])
        xpts = np.linspace(0, L, num_sensors[0])
        ypts = np.linspace(0, w, num_sensors[1])
        xmesh, ymesh = np.meshgrid(xpts, ypts)
        return xmesh, ymesh
    
    def _plot_contour_meanstddiff(self, xmesh, ymesh, values, save_name):

        axis_font = {'fontname':'Arial', 'size':20, 'weight':'normal'}
        labelSize = 18
        cbfontsize = 16

        fig, ax = plt.subplots()
        ax = plt.subplot()
        cs = plt.contourf(xmesh, ymesh, values)
        plt.xlabel("x", **axis_font)
        plt.ylabel("y", **axis_font)
        cb = fig.colorbar(cs)
        cb.ax.tick_params(labelsize=cbfontsize)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(labelSize)

        plt.tight_layout()
        plt.savefig(self.save_dir.joinpath(save_name))
        plt.show()

    def _plot_sample_contours(self, xmesh, ymesh, samples_E, samples_ux, samples_uy, save_name):

        def add_contour(placement, xmesh, ymesh, samples, cbar=True,
                        vmin=None, vmax=None):
            ax = plt.subplot(placement)
            if vmin is not None and vmax is not None:
                levels = MaxNLocator(nbins=9).tick_values(vmin, vmax)
                cs = plt.contourf(xmesh, ymesh, samples.reshape(xmesh.shape),
                                  vmin=vmin, vmax=vmax, levels=levels)
            else:
                cs = plt.contourf(xmesh, ymesh, samples.reshape(xmesh.shape))
            if cbar:
                cb = fig.colorbar(cs)
                if vmin is not None and vmax is not None:
                    levels = MaxNLocator(nbins=9).tick_values(vmin, vmax)

            frame1 = plt.gca()
            frame1.axes.get_xaxis().set_visible(False)
            frame1.axes.get_yaxis().set_visible(False)

        fig, ax = plt.subplots(nrows=3, ncols=3)
        Emin = np.min(samples_E)
        Emax = np.max(samples_E)
        uxmin = 0
        uxmax = 1.6
        uymin = -0.6
        uymax = 0.1
        Emin = None
        Emax = None
        add_contour(331, xmesh, ymesh, samples_E[0, :], True, Emin, Emax)
        add_contour(332, xmesh, ymesh, samples_E[1, :], True, Emin, Emax)
        add_contour(333, xmesh, ymesh, samples_E[2, :], True, Emin, Emax)
        add_contour(334, xmesh, ymesh, samples_ux[0, :], True, uxmin, uxmax)
        add_contour(335, xmesh, ymesh, samples_ux[1, :], True, uxmin, uxmax)
        add_contour(336, xmesh, ymesh, samples_ux[2, :], True, uxmin, uxmax)
        add_contour(337, xmesh, ymesh, samples_uy[0, :], True, uymin, uymax)
        add_contour(338, xmesh, ymesh, samples_uy[1, :], True, uymin, uymax)
        add_contour(339, xmesh, ymesh, samples_uy[2, :], True, uymin, uymax)


        plt.tight_layout()
        plt.savefig(self.save_dir.joinpath(save_name))
        plt.show()
