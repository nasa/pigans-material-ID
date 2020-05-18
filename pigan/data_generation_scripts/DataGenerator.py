#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:14:32 2020

@author: tlewitt
"""

from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
from scipy import interpolate as scipy_interp
from tqdm import tqdm

from disp_field_generator import compute_disps_on_grid
from kle_random_field import KLERandomField



class DataGen():

    def __init__(self, num_test_points_x, num_test_points_y,num_ux_sensors, 
                 num_uy_sensors, num_colloc_pts_x, num_colloc_pts_y, 
                 num_u_bc_sensors, num_sigma_bc_sensors, length, width, 
                 sigma, train_save=None, test_save=None, txt_files_dir=None):
        
        self.NUM_TEST_POINTS = (num_test_points_x, num_test_points_y)
        
        self.NUM_U_SENSORS = (num_ux_sensors,num_uy_sensors)
        
        self.NUM_COLLOC_PTS = (num_colloc_pts_x,num_colloc_pts_y)
        
        self.NUM_U_BC_SENSORS = num_u_bc_sensors
        
        self.NUM_SIGMA_BC_SENSORS = num_sigma_bc_sensors
        
        self.TRAIN_SAVE_NAME = train_save
        
        self.TEST_SAVE_NAME = test_save
        
        self.TXT_FILES_DIR = txt_files_dir
        
        self.L = length
        
        self.w = width
        
        self.SIGMA = sigma

    def generate_hdf5_train_file(self):
        ux_file, uy_file, E_file = self.get_txt_files(True,self.TXT_FILES_DIR)
        self.txt_files_to_hdf5_data_file(ux_file,uy_file,E_file, True)

    def generate_hdf5_test_file(self):

        ux_file, uy_file, E_file = self.get_txt_files(False,self.TXT_FILES_DIR)
        self.txt_files_to_hdf5_data_file(ux_file,uy_file,E_file, False)

    def txt_files_to_hdf5_data_file(self, ux_file, uy_file, E_file, train):

        ux_data = np.genfromtxt(ux_file)
        uy_data = np.genfromtxt(uy_file)
        E_data = np.genfromtxt(E_file)
        if train:
            self._generate_hdf5_data_file_from_data(ux_data, uy_data)
        else:
            self._generate_hdf5_test_file_from_data(ux_data, uy_data, E_data)

    def _generate_hdf5_test_file_from_data(self, ux_test_data, uy_test_data, E_test_data):

        X_test = self._gen_input(self.NUM_TEST_POINTS)
        u_test = self._gen_u_samples(ux_test_data, uy_test_data, X_test)
        E_test = self._gen_E_samples(E_test_data, X_test)
        num_test_samples = ux_test_data.shape[1]-2
        with h5py.File(self.TEST_SAVE_NAME, 'w') as data_file:
            test_group = data_file.create_group('testing')
            test_group.attrs['num_samples'] = num_test_samples
            test_group.attrs['num_sensors'] = self.NUM_TEST_POINTS
            test_group.create_dataset('X_test',
                                      data=tf.cast(X_test, dtype=tf.float32))
            test_group.create_dataset('u_test',
                                      data=tf.cast(u_test, dtype=tf.float32))
            test_group.create_dataset('E_test',
                                      data=tf.cast(E_test, dtype=tf.float32))

    def _generate_hdf5_data_file_from_data(self, ux_train_data, uy_train_data):

        X_u = self._gen_input(self.NUM_U_SENSORS, ignore_left=True)
        X_f = self._gen_input(self.NUM_COLLOC_PTS)

        num_snapshots = ux_train_data.shape[1] - 2

        # Take into account x/y coord columns

        X_ux_bc, ux_bc, X_sigma_bc, sigma_bc, X_sigma_lo, X_sigma_hi = self._gen_bc(self.NUM_U_BC_SENSORS, self.NUM_SIGMA_BC_SENSORS)

        u_train = self._gen_u_samples(ux_train_data, uy_train_data, X_u)

        snapshots = tf.reshape(u_train, [num_snapshots, -1])

        
        with h5py.File(self.TRAIN_SAVE_NAME, 'w') as data_file:
            train_group = data_file.create_group('data')
            bc_group = train_group.create_group('boundary_conditions')

            #Inputs
            train_group.attrs["Num_U_Sensors"] = self.NUM_U_SENSORS
            train_group.attrs["Num_Colloc_Pts"] = self.NUM_COLLOC_PTS
            train_group.create_dataset('X_u',
                                       data=tf.cast(X_u, dtype=tf.float32))
            train_group.create_dataset('X_f',
                                       data=tf.cast(X_f, dtype=tf.float32))

            # Boundary Conditions
            bc_group.create_dataset('X_ux_bc',
                                    data=tf.cast(X_ux_bc, dtype=tf.float32))
            bc_group.create_dataset('X_sigma_bc',
                                    data=tf.cast(X_sigma_bc, dtype=tf.float32))
            bc_group.create_dataset('X_sigma_lo',
                                    data=tf.cast(X_sigma_lo, dtype=tf.float32))
            bc_group.create_dataset('X_sigma_hi',
                                    data=tf.cast(X_sigma_hi, dtype=tf.float32))
            bc_group.create_dataset('ux_bc',
                                    data=tf.cast(ux_bc, dtype=tf.float32))
            bc_group.create_dataset('sigma_bc',
                                    data=tf.cast(sigma_bc, dtype=tf.float32))
            zerosigma = np.zeros(self.NUM_SIGMA_BC_SENSORS)
            zerosigma = np.expand_dims(zerosigma, axis=1)
            bc_group.create_dataset('sigma_zero',
                                    data=tf.cast(zerosigma, dtype=tf.float32))

            train_group.create_dataset('snapshots',
                                       data=tf.cast(snapshots,
                                                    dtype=tf.float32))
    def get_txt_files(self,train, model_dir):
        
        if train:
            SUB_DIR = "train"
        else:
            SUB_DIR = "test"
            
        ux_file = model_dir.joinpath(SUB_DIR,"ux_data.txt")
        uy_file = model_dir.joinpath(SUB_DIR,"uy_data.txt")
        E_file = model_dir.joinpath(SUB_DIR,"E_data.txt")
        
        return ux_file, uy_file, E_file 
    
    def generate_txt_data(self,train,nu,corr_len,alpha,beta,num_snaps,num_test_snaps):
        x_pts = np.linspace(0, self.L, self.NUM_TEST_POINTS[0])
        y_pts = np.linspace(0, self.w, self.NUM_TEST_POINTS[1])
        if train:
            num_samples = num_snaps
        else:
            num_samples = num_test_snaps
        kle_2d = KLERandomField(x_pts, y_pts, corr_len)
        
        #Get Gaussian random fields then transform into log normal fields
        samples_kle = kle_2d.sample_gaussian_random_field(num_samples, 5)
        samples_ln = alpha + beta*np.exp(samples_kle)
        coords = kle_2d.get_xy_grid()
        E_data = np.hstack((coords, samples_ln.T))
        sensor_coords = np.array([(x, y) for y in y_pts for x in x_pts])
        uxdata = np.zeros((len(sensor_coords), 2 + num_samples))
        uxdata[:, :2] = sensor_coords
        uydata = np.copy(uxdata)
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            E_field = E_data[:, i + 2].reshape((-1, 1))
            reshaped_E_data = np.hstack((coords, E_field))
            ux, uy = compute_disps_on_grid(reshaped_E_data, sensor_coords,
                                           self.L, self.w, 40, 40, nu,
                                           1.5, output_paraview=False)
            uxdata[:, i + 2] = ux
            uydata[:, i + 2] = uy
        return uxdata, uydata, E_data


    def save_txt_files(ux, uy, E, train, save_dir):

        if train:
            SUB_DIR = "train/"
        else:
            SUB_DIR = "test/"
        np.savetxt(save_dir.joinpath(SUB_DIR, "ux_data.txt"), ux)
        np.savetxt(save_dir.joinpath(SUB_DIR, "uy_data.txt"), uy)
        np.savetxt(save_dir.joinpath(SUB_DIR, "E_data.txt"), E)


    def _gen_input(self, num_sens, ignore_left = False):
        num_sensors_x, num_sensors_y = num_sens
        x_coords, y_coords = self._get_sensor_coords(self.L, self.w, 
                                                     num_sensors_x, num_sensors_y, ignore_left)
        xy_pairs = self._gen_pairs(x_coords, y_coords)
        return xy_pairs

    def _gen_pairs(self, x_coords, y_coords):

        x_coords = np.expand_dims(x_coords, axis=1)
        y_coords = np.expand_dims(y_coords, axis=1)

        xy_pairs = np.concatenate((x_coords, y_coords), axis=1)
        return xy_pairs

    def _get_sensor_coords(self, L, w, num_sensors_x,num_sensors_y, ignore_left):

        x_sensor_locs = np.linspace(0, L, num_sensors_x)
        #Updated to not include U sensor points on Left Boundary
        if ignore_left:
            x_sensor_locs = x_sensor_locs[1:]
        y_sensor_locs = np.linspace(0, w, num_sensors_y)
        x_grid, y_grid = np.meshgrid(x_sensor_locs, y_sensor_locs)
        return (x_grid.flatten(), y_grid.flatten())

    def _gen_bc(self, num_sensors_bc_ux, num_sensors_bc_sigma):

        x_coords_u_bc = np.zeros(num_sensors_bc_ux)
        y_coords_u_bc = np.linspace(0, self.w, num_sensors_bc_ux)
        xy_pairs_u_bc = self._gen_pairs(x_coords_u_bc, y_coords_u_bc)
        ux_bcs = np.zeros(num_sensors_bc_ux)
        ux_bcs = np.expand_dims(ux_bcs, axis=1)

        x_coords_sigma_bc = np.ones(num_sensors_bc_sigma) * self.L
        y_coords_sigma_bc = np.linspace(0, self.w, num_sensors_bc_sigma)
        xy_pairs_sigma_bc = self._gen_pairs(x_coords_sigma_bc,
                                            y_coords_sigma_bc)
        sigma_bcs = np.ones(num_sensors_bc_sigma) * self.SIGMA
        sigma_bcs = np.expand_dims(sigma_bcs, axis=1)

        x_coords_hi_lo = np.linspace(0, self.L, num_sensors_bc_sigma)
        y_coords_hi = np.ones(num_sensors_bc_sigma) * self.w
        y_coords_lo = np.zeros(num_sensors_bc_sigma)

        xy_pairs_lo = self._gen_pairs(x_coords_hi_lo, y_coords_lo)
        xy_pairs_hi = self._gen_pairs(x_coords_hi_lo, y_coords_hi)

        return xy_pairs_u_bc, ux_bcs, xy_pairs_sigma_bc, sigma_bcs, xy_pairs_lo, xy_pairs_hi

    def _gen_u_samples(self, ux_data, uy_data, X_u):

        u_samples = []

        ux_coords = ux_data[:, 0:2]
        uy_coords = uy_data[:, 0:2]
        ux_data = ux_data[:, 2:]
        uy_data = uy_data[:, 2:]

        num_snapshots = ux_data.shape[1]

        for i in range(num_snapshots):
            u_x = self._interpolate_data(ux_coords, ux_data[:, i], X_u)
            u_y = self._interpolate_data(uy_coords, uy_data[:, i], X_u)
            u_sample = np.vstack((u_x, u_y)).T
            u_samples.append(u_sample)

        u_samples = tf.cast(np.array(u_samples), dtype=tf.float32)

        return u_samples

    def _gen_E_samples(self, E_data, X_E):

        E_samples = []

        E_coords = E_data[:, 0:2]
        E_data = E_data[:, 2:]

        num_snapshots = E_data.shape[1]

        for i in range(num_snapshots):
            E_sample = self._interpolate_data(E_coords, E_data[:, i], X_E)
            E_samples.append(E_sample)

        E_samples = tf.cast(np.array(E_samples), dtype=tf.float32)

        return E_samples

    def _interpolate_data(self, coords, data, X_grid):

        interp = scipy_interp.LinearNDInterpolator(coords, data)
        sample = interp(X_grid)
        return sample
