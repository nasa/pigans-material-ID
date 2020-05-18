import tensorflow as tf


class NoiseSampler():
    def __init__(self, noise_dim):
        self.noise_dim = noise_dim

    def sample_noise(self, num_sensors, batch_size):
        noise = tf.random.normal([1, self.noise_dim])
        noise_arr = tf.tile(noise, [num_sensors, 1])

        for b in range(batch_size - 1):
            noise = tf.random.normal([1, self.noise_dim])
            noise_arr = tf.concat(
                    [noise_arr, tf.tile(noise, [num_sensors, 1])], axis=0)

        return noise_arr
