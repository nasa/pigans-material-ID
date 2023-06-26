"""
Created on Wed Jan 29 15:52:51 2020

@author: tlewitt

"""
from pathlib import Path
import tensorflow as tf
import time
import math
import sys


from pigan.pigan import PIGAN
from pigan.components.generator import Generator
from pigan.components.discriminator import Discriminator
from pigan.components.pde import PDE
from pigan.components.boundary_conditions import BoundaryConditions
from pigan.components.noise_sampler import NoiseSampler
from utilities.general import load_train_dataset

TRAIN_DATA_FILE = Path(sys.argv[1])
BATCH_SIZE = int(sys.argv[2])
NOISE_DIM = int(sys.argv[3])
GEN_ITERS = int(sys.argv[4])
DISC_ITERS = int(sys.argv[5])
LAMBDA = float(sys.argv[6]) # gradpen

PARENT_DIR = '/hpnobackup2/pleser/pigans'
SUB_DIR = f'{TRAIN_DATA_FILE.stem}_b{BATCH_SIZE}_n{NOISE_DIM}_g{GEN_ITERS}_d{DISC_ITERS}_gp{LAMBDA}'

NUM_CHECKPOINTS = 10
TRAINING_STEPS = 50000 #50000
LEARNING_RATE = 1e-4

GEN_INPUT_SHAPE =  NOISE_DIM + 2

train_data, bc_data = load_train_dataset(TRAIN_DATA_FILE)
    
TOTAL_U_SENSORS=train_data['X_u'].shape[0]
DISC_INPUT_SHAPE =int(TOTAL_U_SENSORS * 2)  

# Batch dataset
dataset = tf.data.Dataset.from_tensor_slices(train_data['snapshots'])
batched_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


#Initialize All PIGAN 
noise_sampler = NoiseSampler(NOISE_DIM)
#pde = PDE()
#boundary_conditions = BoundaryConditions(bc_data, noise_sampler)

generator_optimizer = tf.keras.optimizers.legacy.Adam(
                                                learning_rate=LEARNING_RATE,
                                                beta_1=0.1,
                                                beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(
                                                learning_rate=LEARNING_RATE, 
                                                beta_1=0.1,
                                                beta_2=0.9)

generator = Generator(input_shape=GEN_INPUT_SHAPE, 
                      #pde=pde, boundary_conditions=boundary_conditions, 
                      optimizer=generator_optimizer, 
                      noise_sampler=noise_sampler)

discriminator = Discriminator(input_shape=DISC_INPUT_SHAPE, LAMBDA=LAMBDA, 
                              optimizer=discriminator_optimizer, 
                              noise_sampler=noise_sampler)

pigan = PIGAN(generator=generator, discriminator=discriminator,
              parentdir=PARENT_DIR, subdir=SUB_DIR)

step = -1
training_steps_per_chkpt = math.ceil(TRAINING_STEPS / NUM_CHECKPOINTS)
for i in range(NUM_CHECKPOINTS):
    step = pigan.train(inputs=train_data,
                       dataset=batched_dataset,
                       training_steps=training_steps_per_chkpt,
                       generator_iterations=GEN_ITERS, 
                       discriminator_iterations=DISC_ITERS,
                       step=step + 1)

    pigan.save(subdir='chkpt{:04}'.format(i))
pigan.save()
