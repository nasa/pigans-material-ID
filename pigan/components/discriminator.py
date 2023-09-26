import functools
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers


class Discriminator():
    """
    A class that encapsulates the concept of a discriminator that learns to differentiate between samples drawn from the distribution of some dataset, and samples drawn from
    a Generator trained on that same dataset.

    Attributes
    ----------
    discriminator : Sequential
        A Keras Sequential model.
    disc_opt : Optimizer
        A Keras Optimizer to update the generator's weights.
    LAMBDA : float
        Coefficient for the gradient penalty in the loss function.
    noise_sampler : NoiseSampler
        Instance of the NoiseSampler class to generate noise for the generator inputs.
    
    Methods
    -------
    step(inputs, snapshot_batch, generator, batch_size)
        Discriminator training step after which the discriminator's weights are updated.
    _loss(real_output, fake_output, real_snapshots, generated_snapshots, batch_size)
        Calculates the loss for the discriminator using Equation 3 from "Improved Training of Wasserstein GANs" by Gulrajani et al. at a certain training step.
    _gradient_penalty(discriminator, real, fake, batch_size)
        Calculates a gradient penalty based on Equation 3 from "Improved Training of Wasserstein GANs" by Gulrajani et al.
    _model(input_shape)
        Generates a Keras Sequential model.
    save(save_dir)
        Saves the discriminator model.
    """

    def __init__(self, input_shape, LAMBDA, optimizer, noise_sampler):
        """
        Parameters
        ----------
        input_shape : tuple
            Indicates the shape of the input the discriminator model will take. 
        LAMBDA : tf.Tensor
            Coefficient for the gradient penalty in the loss function.
        optimizer : Optimizer
            A Keras Optimizer to update the discriminator's weights.
        noise_sampler : NoiseSampler
            Instance of the NoiseSampler class to generate noise for the generator inputs.
        """
        self.input_shape = input_shape

        self.discriminator = self._model(input_shape)
        self.disc_opt = optimizer
        self.noise_sampler = noise_sampler
        self.LAMBDA = LAMBDA

    def __call__(self):
        pass

    @tf.function
    def step(self, inputs, snapshot_batch, generator, batch_size, train_u=True):
        """Discriminator training step.

        Parameters
        ----------
        inputs : dict
            Dictionary containing inputs for the generator: X_u, X_E (if dataset was generated with E sensors data).

        snapshots_batch : tf.Tensor
            Batch of snapshots from the dataset. Should have shape: [batch size, (number of u sensors * dimensionality) + number of E sensors]

        generator : Generator
            Instance of Generator class. Used to generate samples.

        batch_size : tf.Tensor
            Number of samples in the current batch.

        Returns
        -------
        disc_loss : tf.Tensor
            Discriminator loss at the current step.
        """
        with tf.GradientTape() as disc_tape:
            X_u = inputs['X_u']

            X_u_g = tf.tile(X_u, [batch_size, 1])

            # ω -> noise; shape: [batch_size * num_sensors, NOISE_DIM]
            noise_u = self.noise_sampler.sample_noise(X_u.shape[0], batch_size)

            # u(X, ω) where X is a vector consisting of the input dimensions; shape: 1D -> [batch_size * num_sensors, 1], 2D -> [batch_size * num_sensors, 2]
            generated_u = generator.generator_u(
                tf.concat([X_u_g, noise_u], axis=1), training=True)
            
            # Reshaping for input into Discriminator; shape: 1D -> [batch_size, num_sensors * 1], 2D -> [batch_size, num_sensors * 2]
            generated_snapshots = tf.reshape(generated_u, [batch_size, -1])
            
            fake_output = self.discriminator(generated_snapshots,
                                             training=True)
            real_output = self.discriminator(snapshot_batch, training=True)

            disc_loss = self._loss(real_output, fake_output, snapshot_batch,
                                   generated_snapshots, batch_size)

        gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                        self.discriminator.trainable_variables)
        if train_u:
            self.disc_opt.apply_gradients(zip(gradients_of_discriminator,
                                        self.discriminator.trainable_variables))

        return disc_loss

    def _loss(self, real_output, fake_output, real_snapshots,
              generated_snapshots, batch_size):
        """Calculates the loss for the discriminator using Equation 3 from "Improved Training of Wasserstein GANs" by Gulrajani et al.

        Parameters
        ----------
        real_output : tf.Tensor
            Evaluation of the snapshots from the dataset by the discriminator.

        fake_output : tf.Tensor
            Evaluation of the snapshots from the generator by the discriminator.

        real_snapshots : tf.Tensor
            Batch of snapshots from the dataset. Should have shape: [batch_size, (number of u sensors * dimensionality) + number of E sensors]

        generated_snapshots : tf.Tensor
            Batch of snapshots composed of samples of u and E created by the generator. 
            Has shape: [batch_size, (number of u sensors * dimensionality) + number of E sensors]

        batch_size : tf.Tensor
            Number of samples in the current batch.

        Returns
        -------
        disc_loss : tf.Tensor
            Discriminator loss calculated as the Wasserstein distance between the distribution of the real data and the generated data with a 
            gradient penalty applied to enforce a 1-Lipschitz constraint.  
        """
        real_loss = -tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        wgan_disc_loss = real_loss + fake_loss

        gp = self._gradient_penalty(
            functools.partial(self.discriminator, training=True),
            real_snapshots, generated_snapshots, batch_size)

        print('disc loss = {wgan_disc_loss}')
        print('grad penalty loss (unweighted) = {gp}')
        disc_loss = wgan_disc_loss + (gp * self.LAMBDA)

        return disc_loss

    def _gradient_penalty(self, discriminator, real, fake, batch_size):
        """Calculates a gradient penalty based on Equation 3 from "Improved Training of Wasserstein GANs" by Gulrajani et al.

        Parameters
        ----------
        discriminator : Discriminator
            Instance of Discriminator.

        real : tf.Tensor
            Batch of snapshots from a dataset.

        fake : tf.Tensor
            Batch of snapshots composed of samples of u and E created by the generator. 

        batch_size : tf.Tensor
            Number of samples in the current batch.

        Returns
        gp : tf.Tensor
            Penalty on the gradient norm for random samples sampled uniformly along straight lines between pairs of points sampled from the data 
            distribution and the generated distribution
        """
        real = tf.cast(real, tf.float32)

        real_flat = tf.reshape(real, [batch_size, -1])
        fake_flat = tf.reshape(fake, [batch_size, -1])

        def _interpolate(a, b):
            alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.,
                                      maxval=1.)
            inter = alpha * a + ((1 - alpha) * b)

            return inter

        x = _interpolate(real_flat, fake_flat)

        with tf.GradientTape() as t:
            t.watch(x)
            pred = discriminator(x)

        grad = t.gradient(pred, x)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
        gp = tf.reduce_mean((slopes - 1) ** 2)
        return gp

    def _model(self, input_shape):
        """Creates a Sequential model (linear stack of layers).

        Parameters
        ----------
        input_shape : tuple
            Indicates the shape of the input the model will take.

        Returns
        -------
        model : Sequential
            Keras Sequential model. 
        """
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Flatten())

        model.add(layers.Dense(128, kernel_initializer='glorot_uniform',
                               bias_initializer='zeros'))
        model.add(layers.Activation('tanh'))

        model.add(layers.Dense(128))
        model.add(layers.Activation('tanh'))

        model.add(layers.Dense(128))
        model.add(layers.Activation('tanh'))

        model.add(layers.Dense(128))
        model.add(layers.Activation('tanh'))

        model.add(layers.Dense(1))

        return model

    def save(self, save_dir):
        """Saves the discriminator model (architecture + weights) into an hdf5 file using the save function from the Keras API.

        Parameters
        ----------
        save_dir : string
            Path to the directory where the model will be saved.
        """
        self.discriminator.save(save_dir.joinpath('discriminator.h5'))

    def load(self, model_dir):
        """Loads the discriminator model (architecture + weights) from an hdf5 file using the load_model function from the Keras API.

        Parameters
        ----------
        model_dir : string
            Path to the directory where the model is saved.
        """
        self.discriminator = tf.keras.models.load_model(
            model_dir.joinpath('discriminator.h5'), compile=False)

