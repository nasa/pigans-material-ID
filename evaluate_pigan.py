#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:15:26 2020

@author: tlewitt
"""
from pathlib import Path

from pigan.post_processing_scripts.PostProcessor import PostProcessor
from pigan.pigan import PIGAN
from utilities.general import load_test_dataset



#----------Input Parameters---------------#
#File Locatiions
MODEL_DIR = Path("data/models_paper/")
TEST_FILE = Path("data/dataset_test.hdf5")

#Toggles saving generated data
SAVE_RESULTS = True

#Number of Prediction Samples
NUM_PREDICT_SAMPLES = 1000



#Load Trained PIGAN Model
trained_model = PIGAN()
trained_model.load(MODEL_DIR)

#Load Test Data
x_test, E_test, ux_test, uy_test = load_test_dataset(TEST_FILE)

#Generate Data From Trained Model
E_gen, ux_gen, uy_gen = trained_model.generate(x_test,
                                               NUM_PREDICT_SAMPLES,
                                               SAVE_RESULTS)

post_processor = PostProcessor(x_test, E_test, E_gen, ux_test, ux_gen,
                               uy_test, uy_gen)

post_processor.create_paper_graphs()
