import sys
import time
from pathlib import Path
import time
import h5py
import tensorflow as tf
from tqdm import tqdm

from pigan.components.boundary_conditions import BoundaryConditions
from pigan.components.discriminator import Discriminator
from pigan.components.generator import Generator
from pigan.components.noise_sampler import NoiseSampler
from pigan.components.pde import PDE
from utilities.general import save_log, save_samples

class PIGAN():
    def __init__(self, generator=None, discriminator=None):
        self.generator = generator
        self.discriminator = discriminator
        self.gen_log = []
        self.disc_log = []
        self.time_log = []

        self.save_dir = Path("data/")

    def train(self, inputs, dataset, training_steps, generator_iterations, 
              discriminator_iterations):
        log_dir = Path("data/tensorboard/")
        writer_path = log_dir / time.strftime("%b_%d_%H_%M_%s/",time.gmtime())
        writer = tf.summary.create_file_writer(str(writer_path))
        
        train_start_time = time.time()
        for step in tqdm(range(training_steps), desc='Training'):
            start = time.time()
            for batch in dataset:
                batch_size = batch.shape[0]
                for i in range(discriminator_iterations):
                    disc_loss = self.discriminator.step(inputs, batch,
                                                        self.generator,
                                                        batch_size=batch_size)
                    self.disc_log.append(
                        {'step': (step * discriminator_iterations) + i + 1,
                         'disc_loss': disc_loss.numpy()})
                    with writer.as_default():
                        tf.summary.scalar('Discriminator Loss',disc_loss.numpy(), 
                                           step = (step * discriminator_iterations) + i + 1)

                for i in range(generator_iterations):
                    gen_loss, pde_loss, bc_loss = \
                        self.generator.step(inputs, self.discriminator,
                                            batch_size=batch.shape[0])
                    self.gen_log.append(
                        {'step': (step * generator_iterations) + i + 1,
                         'gen_loss': gen_loss.numpy(),
                         'pde_loss': pde_loss.numpy(),
                         'bc_loss': bc_loss.numpy()})
                    with writer.as_default():
                        tf.summary.scalar('Generator Loss',gen_loss.numpy(), 
                                           step = (step * generator_iterations) + i + 1)
                        tf.summary.scalar('PDE Loss',pde_loss.numpy(), 
                                           step = (step * generator_iterations) + i + 1)
                        tf.summary.scalar('BC Loss',bc_loss.numpy(), 
                                           step = (step * generator_iterations) + i + 1)


            if (step + 1) % 50 == 0:
                elapsed_time = time.time() - train_start_time
                sec_per_step = elapsed_time / step
                mins_left = ((training_steps - step) * sec_per_step)
                tf.print("\nStep # ", step, "/", training_steps,
                         output_stream=sys.stdout)
                tf.print("Current time:", elapsed_time, " time left:",
                         mins_left, output_stream=sys.stdout)
                tf.print("Discriminator Loss: ", disc_loss,
                         output_stream=sys.stdout)
                tf.print("Generator Loss: ", gen_loss,
                         output_stream=sys.stdout)
                tf.print("PDE Loss: ", pde_loss, output_stream=sys.stdout)
                tf.print("BC Loss: ", bc_loss, output_stream=sys.stdout)


            elapsed_time =  time.time() - start
            self.time_log.append({'step': step + 1,
                                  'elapsed_time': elapsed_time})
            with writer.as_default():
                tf.summary.scalar('Time Per Step', elapsed_time, step = step + 1)


    def save(self, logs=True,debug=False):
        if debug:
            self.save_dir = self.save_dir.joinpath("debug/")
        model_save_dir = self.save_dir.joinpath("models/")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        self.generator.save(model_save_dir)
        self.discriminator.save(model_save_dir)

        tf.print("Saving PIGAN in: " + str(model_save_dir), output_stream=sys.stdout)

        with h5py.File(model_save_dir.joinpath('pigan_settings.hdf5'),
                       'w') as settings_file:
            settings_file.attrs['gen_input_shape'] = self.generator.input_shape
            settings_file.attrs['disc_input_shape'] = \
                self.discriminator.input_shape
            settings_file.attrs['LAMBDA'] = self.discriminator.LAMBDA
            settings_file.attrs['learning_rate'] = self.generator.gen_opt.learning_rate.numpy()

        if logs:
            log_dir = self.save_dir.joinpath('logs/')
            log_dir.mkdir(parents=True, exist_ok=True)
            save_log(self.disc_log, 'disc_loss_per_step',
                     self.disc_log[0].keys(), log_dir)
            save_log(self.gen_log, 'gen_loss_per_step', self.gen_log[0].keys(),
                     log_dir)
            save_log(self.time_log, 'time_per_step', self.time_log[0].keys(),
                     log_dir)

    def generate(self, test_data, num_samples, save_results):
        """Calls the generate function of the PIGAN's generator to generate a
        number of a samples.

        Parameters
        ----------
        X : array_like
            Array of points/coordinates with shape [number of
            points/coordinates, dimensionality],
            e.g., [10, 1] -> 1D, [10, 2] -> 2D

        num_smaples : int
            Number of samples to generate.

        save_dir : string
            Path to the directory where the results will be saved.

        Returns
        -------
        array
            An array with shape: [num_samples, number of points/coordinates,
            dimensionality]
        """
        gen_U, gen_E = self.generator.generate(test_data, num_samples)
        E = gen_E[:, :, 0].numpy()
        ux = gen_U[:, :, 0].numpy()
        uy = gen_U[:, :, 1].numpy()
        if save_results:
            save_samples(gen_E, gen_U, test_data,Path("data/generated_snapshots.hdf5"))
        return E, ux, uy
    
    def load(self, model_dir):

        with h5py.File(model_dir.joinpath('pigan_settings.hdf5'),
                       'r') as settings_file:

            noise_sampler = NoiseSampler(int(settings_file.attrs['gen_input_shape'])-2)
            pde = PDE()
            boundary_conditions = BoundaryConditions(None, noise_sampler)
            LEARNING_RATE = settings_file.attrs['learning_rate']
            generator_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE,
                                                           beta_1=0.0,
                                                           beta_2=0.9)
            discriminator_optimizer = tf.keras.optimizers.Adam(
                lr=LEARNING_RATE, beta_1=0.0, beta_2=0.9)
            self.generator = Generator(
                input_shape=int(settings_file.attrs['gen_input_shape']),
                pde=pde, boundary_conditions=boundary_conditions,
                optimizer=generator_optimizer, noise_sampler=noise_sampler)

            self.discriminator = Discriminator(
                input_shape=int(settings_file.attrs['disc_input_shape']),
                LAMBDA=settings_file.attrs['LAMBDA'],
                optimizer=discriminator_optimizer, noise_sampler=noise_sampler)

            self.generator.load(model_dir)
            self.discriminator.load(model_dir)
