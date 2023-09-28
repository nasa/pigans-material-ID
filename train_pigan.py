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
PDE_WEIGHT = float(sys.argv[7])
BC_WEIGHT = float(sys.argv[8])
LEARNING_RATE = float(sys.argv[9])
E_WEIGHT = float(sys.argv[10])

PARENT_DIR = '/hpnobackup2/pleser/pigans'
SUB_DIR = f'{TRAIN_DATA_FILE.stem}_b{BATCH_SIZE}_n{NOISE_DIM}_g{GEN_ITERS}_d{DISC_ITERS}_gp{LAMBDA}_pw{PDE_WEIGHT:.2e}_bc{BC_WEIGHT:.2e}_lr{LEARNING_RATE:.2e}_E{E_WEIGHT:.2e}'

DU_SCALE_FACTOR = 1/1000
NUM_CHECKPOINTS = 20
TRAINING_STEPS = 1000000
MAX_U_TRAIN_STEPS = 2E6
ESTIMATE_NOISE = True

GEN_INPUT_SHAPE =  NOISE_DIM + 2

train_data, bc_data = load_train_dataset(TRAIN_DATA_FILE)
    
TOTAL_U_SENSORS=train_data['X_u'].shape[0]
DISC_INPUT_SHAPE =int(TOTAL_U_SENSORS * 2)  

# Batch dataset
dataset = tf.data.Dataset.from_tensor_slices(train_data['snapshots'])
batched_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


#Initialize All PIGAN 
noise_sampler = NoiseSampler(NOISE_DIM)
pde = PDE(scale_factor=DU_SCALE_FACTOR)
boundary_conditions = BoundaryConditions(bc_data,
                                         noise_sampler,
                                         DU_SCALE_FACTOR)

generator_optimizer = tf.keras.optimizers.legacy.Adam(
                                                learning_rate=LEARNING_RATE,
                                                beta_1=0.1,
                                                beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(
                                                learning_rate=LEARNING_RATE, 
                                                beta_1=0.1,
                                                beta_2=0.9)

generator = Generator(input_shape=GEN_INPUT_SHAPE, 
                      pde=pde,
                      boundary_conditions=boundary_conditions, 
                      optimizer=generator_optimizer, 
                      noise_sampler=noise_sampler,
                      pde_weight=PDE_WEIGHT,
                      bc_weight=BC_WEIGHT,
                      E_weight=E_WEIGHT,
                      estimate_noise=ESTIMATE_NOISE)

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
                       step=step + 1,
                       max_u_train_steps=MAX_U_TRAIN_STEPS)

    pigan.save(subdir='chkpt{:04}'.format(i))
pigan.save()
