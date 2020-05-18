import numpy as np

class KLERandomField():

    def __init__(self, x_grid, y_grid, corr_len):

        xy = [(x_, y_) for y_ in y_grid for x_ in x_grid]
        self._num_x = len(x_grid)
        self._num_y = len(y_grid)
        self._xy_grid = np.array(xy)
        self._corr_len = corr_len
        self._calculate_eig_vecs_vals(self._xy_grid, corr_len)

    def get_xy_grid(self):

        return self._xy_grid

    def _calculate_eig_vecs_vals(self, xy_grid, corr_len):

        corrmat = np.zeros((len(xy_grid), len(xy_grid)))
        for i in range(len(xy_grid)):
            for j in range(i, len(xy_grid)):
                corrmat[i, j] = main_corr_func(xy_grid[i], xy_grid[j], corr_len)
                corrmat[j, i] = corrmat[i, j]
        w, v = np.linalg.eig(corrmat)
        self._corrmat = corrmat
        self._eig_vals = w
        self._eig_vecs = v

    def sample_gaussian_random_field(self, num_samples, num_comps):

        samples = np.zeros((num_samples, len(self._xy_grid)))

        for i in range(num_samples):
            random_vars = np.random.normal(size=num_comps)
            samples[i, :] = self.generate_kle_sample(num_comps, random_vars)

        return samples

    def generate_kle_sample(self, num_comps, random_vars):

        sample = np.zeros(len(self._xy_grid))
        real_eigs = self._eig_vals.real

        for i in range(num_comps):
            rv_i = random_vars[i]
            sample += rv_i * np.sqrt(real_eigs[i]) * self._eig_vecs[:, i].real

        return sample

def main_corr_func(xy1, xy2, corr_len):
    dist = np.linalg.norm(xy1-xy2)
    corr = np.exp(-(dist/corr_len)**2.0)
    return corr
