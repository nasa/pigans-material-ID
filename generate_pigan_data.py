#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:15:26 2020

@author: tlewitt
"""
from pathlib import Path
from pigan.data_generation_scripts.DataGenerator import DataGen


#------Input Parameters------------------#
#Dimensions of Plate
LENGTH = 1.0
WIDTH = 1.0
# Force pulling on the plate
SIGMA = 1.5
# Number of Points for Testing Grid 
NUM_TEST_POINTS_X = 25
NUM_TEST_POINTS_Y = 25

#Number of Points for Sensor Grid
NUM_UX_SENSORS = 10
NUM_UY_SENSORS = 10

#Number of Points for PDE Evaluation Grid
NUM_COLLOC_PTS_X = 10
NUM_COLLOC_PTS_Y = 10

#Number of Points for Boundary Condition Evaulations
NUM_U_BC_SENSORS = 10
NUM_SIGMA_BC_SENSORS = 10

#File locations
TRAIN_SAVE_NAME = Path("data/dataset_train.hdf5")
TEST_SAVE_NAME = Path("data/dataset_test.hdf5")
TXT_FILES_DIR = Path("data/txt_files/")

data_gen = DataGen(NUM_TEST_POINTS_X, NUM_TEST_POINTS_Y, NUM_UX_SENSORS,
                   NUM_UY_SENSORS, NUM_COLLOC_PTS_X, NUM_COLLOC_PTS_Y,
                   NUM_U_BC_SENSORS, NUM_SIGMA_BC_SENSORS, LENGTH, WIDTH,
                   SIGMA, TRAIN_SAVE_NAME, TEST_SAVE_NAME, TXT_FILES_DIR)

data_gen.generate_hdf5_train_file()
data_gen.generate_hdf5_test_file()
